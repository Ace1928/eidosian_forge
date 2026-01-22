import collections
import contextlib
import functools
import sys
import threading
import fasteners
import futurist
from kazoo import exceptions as k_exceptions
from kazoo.protocol import paths as k_paths
from kazoo.protocol import states as k_states
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow.conductors import base as c_base
from taskflow import exceptions as excp
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
class ZookeeperJobBoard(base.NotifyingJobBoard):
    """A jobboard backed by `zookeeper`_.

    Powered by the `kazoo <https://kazoo.readthedocs.io/en/latest/>`_ library.

    This jobboard creates *sequenced* persistent znodes in a directory in
    zookeeper and uses zookeeper watches to notify other jobboards of
    jobs which were posted using the :meth:`.post` method (this creates a
    znode with job contents/details encoded in `json`_). The users of these
    jobboard(s) (potentially on disjoint sets of machines) can then iterate
    over the available jobs and decide if they want
    to attempt to claim one of the jobs they have iterated over. If so they
    will then attempt to contact zookeeper and they will attempt to create a
    ephemeral znode using the name of the persistent znode + ".lock" as a
    postfix. If the entity trying to use the jobboard to :meth:`.claim` the
    job is able to create a ephemeral znode with that name then it will be
    allowed (and expected) to perform whatever *work* the contents of that
    job described. Once the claiming entity is finished the ephemeral znode
    and persistent znode will be deleted (if successfully completed) in a
    single transaction. If the claiming entity is not successful (or the
    entity that claimed the znode dies) the ephemeral znode will be
    released (either manually by using :meth:`.abandon` or automatically by
    zookeeper when the ephemeral node and associated session is deemed to
    have been lost).

    Do note that the creation of a kazoo client is achieved
    by :py:func:`~taskflow.utils.kazoo_utils.make_client` and the transfer
    of this jobboard configuration to that function to make a
    client may happen at ``__init__`` time. This implies that certain
    parameters from this jobboard configuration may be provided to
    :py:func:`~taskflow.utils.kazoo_utils.make_client` such
    that if a client was not provided by the caller one will be created
    according to :py:func:`~taskflow.utils.kazoo_utils.make_client`'s
    specification

    .. _zookeeper: http://zookeeper.apache.org/
    .. _json: https://json.org/
    """
    MIN_ZK_VERSION = (3, 4, 0)
    LOCK_POSTFIX = '.lock'
    TRASH_FOLDER = '.trash'
    ENTITY_FOLDER = '.entities'
    JOB_PREFIX = 'job'
    DEFAULT_PATH = '/taskflow/jobs'
    STATE_HISTORY_LENGTH = 2
    '\n    Number of prior state changes to keep a history of, mainly useful\n    for history tracking and debugging connectivity issues.\n    '
    NO_FETCH_STATES = (k_states.KazooState.LOST, k_states.KazooState.SUSPENDED)
    '\n    Client states underwhich we return empty lists from fetching routines,\n    during these states the underlying connection either is being recovered\n    or may be recovered (aka, it has not full disconnected).\n    '

    def __init__(self, name, conf, client=None, persistence=None, emit_notifications=True):
        super(ZookeeperJobBoard, self).__init__(name, conf)
        if client is not None:
            self._client = client
            self._owned = False
        else:
            self._client = kazoo_utils.make_client(self._conf)
            self._owned = True
        path = str(conf.get('path', self.DEFAULT_PATH))
        if not path:
            raise ValueError('Empty zookeeper path is disallowed')
        if not k_paths.isabs(path):
            raise ValueError('Zookeeper path must be absolute')
        self._path = path
        self._trash_path = self._path.replace(k_paths.basename(self._path), self.TRASH_FOLDER)
        self._entity_path = self._path.replace(k_paths.basename(self._path), self.ENTITY_FOLDER)
        self._persistence = persistence
        self._known_jobs = {}
        self._job_cond = threading.Condition()
        self._open_close_lock = threading.RLock()
        self._client.add_listener(self._state_change_listener)
        self._bad_paths = frozenset([path])
        self._job_watcher = None
        self._job_base = k_paths.join(path, self.JOB_PREFIX)
        self._worker = None
        self._emit_notifications = bool(emit_notifications)
        self._connected = False
        self._suspended = False
        self._closing = False
        self._last_states = collections.deque(maxlen=self.STATE_HISTORY_LENGTH)

    def _try_emit(self, state, details):
        worker = self._worker
        if worker is None or not self._emit_notifications:
            return
        try:
            worker.submit(self.notifier.notify, state, details)
        except RuntimeError:
            pass

    @property
    def path(self):
        """Path where all job znodes will be stored."""
        return self._path

    @property
    def trash_path(self):
        """Path where all trashed job znodes will be stored."""
        return self._trash_path

    @property
    def entity_path(self):
        """Path where all conductor info znodes will be stored."""
        return self._entity_path

    @property
    def job_count(self):
        return len(self._known_jobs)

    def _fetch_jobs(self, ensure_fresh=False):
        try:
            last_state = self._last_states[0]
        except IndexError:
            last_state = None
        if last_state in self.NO_FETCH_STATES:
            if last_state == k_states.KazooState.LOST and self._known_jobs:
                self._on_job_posting([], delayed=False)
            return []
        else:
            if ensure_fresh:
                self._force_refresh()
            with self._job_cond:
                return sorted(self._known_jobs.values())

    def _force_refresh(self):
        try:
            maybe_children = self._client.get_children(self.path)
            self._on_job_posting(maybe_children, delayed=False)
        except self._client.handler.timeout_exception:
            excp.raise_with_cause(excp.JobFailure, 'Refreshing failure, operation timed out')
        except k_exceptions.SessionExpiredError:
            excp.raise_with_cause(excp.JobFailure, 'Refreshing failure, session expired')
        except k_exceptions.NoNodeError:
            pass
        except k_exceptions.KazooException:
            excp.raise_with_cause(excp.JobFailure, 'Refreshing failure, internal error')

    def iterjobs(self, only_unclaimed=False, ensure_fresh=False):
        board_removal_func = lambda job: self._remove_job(job.path)
        return base.JobBoardIterator(self, LOG, only_unclaimed=only_unclaimed, ensure_fresh=ensure_fresh, board_fetch_func=self._fetch_jobs, board_removal_func=board_removal_func)

    def _remove_job(self, path):
        if path not in self._known_jobs:
            return False
        with self._job_cond:
            job = self._known_jobs.pop(path, None)
        if job is not None:
            LOG.debug("Removed job that was at path '%s'", path)
            self._try_emit(base.REMOVAL, details={'job': job})
            return True
        else:
            return False

    def _process_child(self, path, request, quiet=True):
        """Receives the result of a child data fetch request."""
        job = None
        try:
            raw_data, node_stat = request.get()
            job_data = misc.decode_json(raw_data)
            job_created_on = misc.millis_to_datetime(node_stat.ctime)
            try:
                job_priority = job_data['priority']
                job_priority = base.JobPriority.convert(job_priority)
            except KeyError:
                job_priority = base.JobPriority.NORMAL
            job_uuid = job_data['uuid']
            job_name = job_data['name']
        except (ValueError, TypeError, KeyError):
            with excutils.save_and_reraise_exception(reraise=not quiet):
                LOG.warning('Incorrectly formatted job data found at path: %s', path, exc_info=True)
        except self._client.handler.timeout_exception:
            with excutils.save_and_reraise_exception(reraise=not quiet):
                LOG.warning('Operation timed out fetching job data from from path: %s', path, exc_info=True)
        except k_exceptions.SessionExpiredError:
            with excutils.save_and_reraise_exception(reraise=not quiet):
                LOG.warning('Session expired fetching job data from path: %s', path, exc_info=True)
        except k_exceptions.NoNodeError:
            LOG.debug('No job node found at path: %s, it must have disappeared or was removed', path)
        except k_exceptions.KazooException:
            with excutils.save_and_reraise_exception(reraise=not quiet):
                LOG.warning('Internal error fetching job data from path: %s', path, exc_info=True)
        else:
            with self._job_cond:
                if path not in self._known_jobs:
                    job = ZookeeperJob(self, job_name, self._client, path, backend=self._persistence, uuid=job_uuid, book_data=job_data.get('book'), details=job_data.get('details', {}), created_on=job_created_on, priority=job_priority)
                    self._known_jobs[path] = job
                    self._job_cond.notify_all()
        if job is not None:
            self._try_emit(base.POSTED, details={'job': job})

    def _on_job_posting(self, children, delayed=True):
        LOG.debug('Got children %s under path %s', children, self.path)
        child_paths = []
        for c in children:
            if c.endswith(self.LOCK_POSTFIX) or not c.startswith(self.JOB_PREFIX):
                continue
            child_paths.append(k_paths.join(self.path, c))
        investigate_paths = []
        pending_removals = []
        with self._job_cond:
            for path in self._known_jobs.keys():
                if path not in child_paths:
                    pending_removals.append(path)
        for path in child_paths:
            if path in self._bad_paths:
                continue
            if path in self._known_jobs:
                continue
            if path not in investigate_paths:
                investigate_paths.append(path)
        if pending_removals:
            with self._job_cond:
                am_removed = 0
                try:
                    for path in pending_removals:
                        am_removed += int(self._remove_job(path))
                finally:
                    if am_removed:
                        self._job_cond.notify_all()
        for path in investigate_paths:
            request = self._client.get_async(path)
            if delayed:
                request.rawlink(functools.partial(self._process_child, path))
            else:
                self._process_child(path, request, quiet=False)

    def post(self, name, book=None, details=None, priority=base.JobPriority.NORMAL):
        job_priority = base.JobPriority.convert(priority)
        job_uuid = uuidutils.generate_uuid()
        job_posting = base.format_posting(job_uuid, name, book=book, details=details, priority=job_priority)
        raw_job_posting = misc.binary_encode(jsonutils.dumps(job_posting))
        with self._wrap(job_uuid, None, fail_msg_tpl='Posting failure: %s', ensure_known=False):
            job_path = self._client.create(self._job_base, value=raw_job_posting, sequence=True, ephemeral=False)
            job = ZookeeperJob(self, name, self._client, job_path, backend=self._persistence, book=book, details=details, uuid=job_uuid, book_data=job_posting.get('book'), priority=job_priority)
            with self._job_cond:
                self._known_jobs[job_path] = job
                self._job_cond.notify_all()
            self._try_emit(base.POSTED, details={'job': job})
            return job

    @base.check_who
    def claim(self, job, who):

        def _unclaimable_try_find_owner(cause):
            try:
                owner = self.find_owner(job)
            except Exception:
                owner = None
            if owner:
                message = "Job %s already claimed by '%s'" % (job.uuid, owner)
            else:
                message = 'Job %s already claimed' % job.uuid
            excp.raise_with_cause(excp.UnclaimableJob, message, cause=cause)
        with self._wrap(job.uuid, job.path, fail_msg_tpl='Claiming failure: %s'):
            value = jsonutils.dumps({'owner': who})
            job_data, job_stat = self._client.get(job.path)
            txn = self._client.transaction()
            txn.check(job.path, version=job_stat.version)
            txn.create(job.lock_path, value=misc.binary_encode(value), ephemeral=True)
            try:
                kazoo_utils.checked_commit(txn)
            except k_exceptions.NodeExistsError as e:
                _unclaimable_try_find_owner(e)
            except kazoo_utils.KazooTransactionException as e:
                if len(e.failures) < 2:
                    raise
                else:
                    if isinstance(e.failures[0], k_exceptions.NoNodeError):
                        excp.raise_with_cause(excp.NotFound, 'Job %s not found to be claimed' % job.uuid, cause=e.failures[0])
                    if isinstance(e.failures[1], k_exceptions.NodeExistsError):
                        _unclaimable_try_find_owner(e.failures[1])
                    else:
                        excp.raise_with_cause(excp.UnclaimableJob, 'Job %s claim failed due to transaction not succeeding' % job.uuid, cause=e)

    @contextlib.contextmanager
    def _wrap(self, job_uuid, job_path, fail_msg_tpl='Failure: %s', ensure_known=True):
        if job_path:
            fail_msg_tpl += ' (%s)' % job_path
        if ensure_known:
            if not job_path:
                raise ValueError('Unable to check if %r is a known path' % job_path)
            if job_path not in self._known_jobs:
                fail_msg_tpl += ', unknown job'
                raise excp.NotFound(fail_msg_tpl % job_uuid)
        try:
            yield
        except self._client.handler.timeout_exception:
            fail_msg_tpl += ', operation timed out'
            excp.raise_with_cause(excp.JobFailure, fail_msg_tpl % job_uuid)
        except k_exceptions.SessionExpiredError:
            fail_msg_tpl += ', session expired'
            excp.raise_with_cause(excp.JobFailure, fail_msg_tpl % job_uuid)
        except k_exceptions.NoNodeError:
            fail_msg_tpl += ', unknown job'
            excp.raise_with_cause(excp.NotFound, fail_msg_tpl % job_uuid)
        except k_exceptions.KazooException:
            fail_msg_tpl += ', internal error'
            excp.raise_with_cause(excp.JobFailure, fail_msg_tpl % job_uuid)

    def find_owner(self, job):
        with self._wrap(job.uuid, job.path, fail_msg_tpl='Owner query failure: %s', ensure_known=False):
            try:
                self._client.sync(job.lock_path)
                raw_data, _lock_stat = self._client.get(job.lock_path)
                data = misc.decode_json(raw_data)
                owner = data.get('owner')
            except k_exceptions.NoNodeError:
                owner = None
            return owner

    def _get_owner_and_data(self, job):
        lock_data, lock_stat = self._client.get(job.lock_path)
        job_data, job_stat = self._client.get(job.path)
        return (misc.decode_json(lock_data), lock_stat, misc.decode_json(job_data), job_stat)

    def register_entity(self, entity):
        entity_type = entity.kind
        if entity_type == c_base.Conductor.ENTITY_KIND:
            entity_path = k_paths.join(self.entity_path, entity_type)
            try:
                self._client.ensure_path(entity_path)
                self._client.create(k_paths.join(entity_path, entity.name), value=misc.binary_encode(jsonutils.dumps(entity.to_dict())), ephemeral=True)
            except k_exceptions.NodeExistsError:
                pass
            except self._client.handler.timeout_exception:
                excp.raise_with_cause(excp.JobFailure, 'Can not register entity %s under %s, operation timed out' % (entity.name, entity_path))
            except k_exceptions.SessionExpiredError:
                excp.raise_with_cause(excp.JobFailure, 'Can not register entity %s under %s, session expired' % (entity.name, entity_path))
            except k_exceptions.KazooException:
                excp.raise_with_cause(excp.JobFailure, 'Can not register entity %s under %s, internal error' % (entity.name, entity_path))
        else:
            raise excp.NotImplementedError("Not implemented for other entity type '%s'" % entity_type)

    @base.check_who
    def consume(self, job, who):
        with self._wrap(job.uuid, job.path, fail_msg_tpl='Consumption failure: %s'):
            try:
                owner_data = self._get_owner_and_data(job)
                lock_data, lock_stat, data, data_stat = owner_data
            except k_exceptions.NoNodeError:
                excp.raise_with_cause(excp.NotFound, 'Can not consume a job %s which we can not determine the owner of' % job.uuid)
            if lock_data.get('owner') != who:
                raise excp.JobFailure('Can not consume a job %s which is not owned by %s' % (job.uuid, who))
            txn = self._client.transaction()
            txn.delete(job.lock_path, version=lock_stat.version)
            txn.delete(job.path, version=data_stat.version)
            kazoo_utils.checked_commit(txn)
            self._remove_job(job.path)

    @base.check_who
    def abandon(self, job, who):
        with self._wrap(job.uuid, job.path, fail_msg_tpl='Abandonment failure: %s'):
            try:
                owner_data = self._get_owner_and_data(job)
                lock_data, lock_stat, data, data_stat = owner_data
            except k_exceptions.NoNodeError:
                excp.raise_with_cause(excp.NotFound, 'Can not abandon a job %s which we can not determine the owner of' % job.uuid)
            if lock_data.get('owner') != who:
                raise excp.JobFailure('Can not abandon a job %s which is not owned by %s' % (job.uuid, who))
            txn = self._client.transaction()
            txn.delete(job.lock_path, version=lock_stat.version)
            kazoo_utils.checked_commit(txn)

    @base.check_who
    def trash(self, job, who):
        with self._wrap(job.uuid, job.path, fail_msg_tpl='Trash failure: %s'):
            try:
                owner_data = self._get_owner_and_data(job)
                lock_data, lock_stat, data, data_stat = owner_data
            except k_exceptions.NoNodeError:
                excp.raise_with_cause(excp.NotFound, 'Can not trash a job %s which we can not determine the owner of' % job.uuid)
            if lock_data.get('owner') != who:
                raise excp.JobFailure('Can not trash a job %s which is not owned by %s' % (job.uuid, who))
            trash_path = job.path.replace(self.path, self.trash_path)
            value = misc.binary_encode(jsonutils.dumps(data))
            txn = self._client.transaction()
            txn.create(trash_path, value=value)
            txn.delete(job.lock_path, version=lock_stat.version)
            txn.delete(job.path, version=data_stat.version)
            kazoo_utils.checked_commit(txn)

    def _state_change_listener(self, state):
        if self._last_states:
            LOG.debug("Kazoo client has changed to state '%s' from prior states '%s'", state, self._last_states)
        else:
            LOG.debug("Kazoo client has changed to state '%s' (from its initial/uninitialized state)", state)
        self._last_states.appendleft(state)
        if state == k_states.KazooState.LOST:
            self._connected = False
            if not self._closing:
                LOG.warning('Connection to zookeeper has been lost')
        elif state == k_states.KazooState.SUSPENDED:
            LOG.warning('Connection to zookeeper has been suspended')
            self._suspended = True
        elif self._suspended:
            self._suspended = False

    def wait(self, timeout=None):
        watch = timeutils.StopWatch(duration=timeout)
        watch.start()
        with self._job_cond:
            while True:
                if not self._known_jobs:
                    if watch.expired():
                        raise excp.NotFound('Expired waiting for jobs to arrive; waited %s seconds' % watch.elapsed())
                    self._job_cond.wait(watch.leftover(return_none=True))
                else:
                    curr_jobs = self._fetch_jobs()
                    fetch_func = lambda ensure_fresh: curr_jobs
                    removal_func = lambda a_job: self._remove_job(a_job.path)
                    return base.JobBoardIterator(self, LOG, board_fetch_func=fetch_func, board_removal_func=removal_func)

    @property
    def connected(self):
        return self._connected and self._client.connected

    @fasteners.locked(lock='_open_close_lock')
    def close(self):
        if self._owned:
            LOG.debug('Stopping client')
            self._closing = True
            kazoo_utils.finalize_client(self._client)
        if self._worker is not None:
            LOG.debug('Shutting down the notifier')
            self._worker.shutdown()
            self._worker = None
        with self._job_cond:
            self._known_jobs.clear()
        LOG.debug('Stopped & cleared local state')
        self._connected = False
        self._last_states.clear()

    @fasteners.locked(lock='_open_close_lock')
    def connect(self, timeout=10.0):

        def try_clean():
            try:
                self.close()
            except k_exceptions.KazooException:
                LOG.exception('Failed cleaning-up after post-connection initialization failed')
        try:
            if timeout is not None:
                timeout = float(timeout)
            self._client.start(timeout=timeout)
            self._closing = False
        except (self._client.handler.timeout_exception, k_exceptions.KazooException):
            excp.raise_with_cause(excp.JobFailure, 'Failed to connect to zookeeper')
        try:
            if strutils.bool_from_string(self._conf.get('check_compatible'), default=True):
                kazoo_utils.check_compatible(self._client, self.MIN_ZK_VERSION)
            if self._worker is None and self._emit_notifications:
                self._worker = futurist.ThreadPoolExecutor(max_workers=1)
            self._client.ensure_path(self.path)
            self._client.ensure_path(self.trash_path)
            if self._job_watcher is None:
                self._job_watcher = watchers.ChildrenWatch(self._client, self.path, func=self._on_job_posting, allow_session_lost=True)
            self._connected = True
        except excp.IncompatibleVersion:
            with excutils.save_and_reraise_exception():
                try_clean()
        except (self._client.handler.timeout_exception, k_exceptions.KazooException):
            exc_type, exc, exc_tb = sys.exc_info()
            try:
                try_clean()
                excp.raise_with_cause(excp.JobFailure, 'Failed to do post-connection initialization', cause=exc)
            finally:
                del (exc_type, exc, exc_tb)