import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
class ThreadGroupManager(object):

    def __init__(self):
        super(ThreadGroupManager, self).__init__()
        self.groups = {}
        self.msg_queues = collections.defaultdict(list)
        self.add_timer(None, self._service_task)

    def _service_task(self):
        """Dummy task which gets queued on the service.Service threadgroup.

        Without this, service.Service sees nothing running i.e has nothing to
        wait() on, so the process exits. This could also be used to trigger
        periodic non-stack-specific housekeeping tasks.
        """
        pass

    def _serialize_profile_info(self):
        prof = profiler.get()
        trace_info = None
        if prof:
            trace_info = {'hmac_key': prof.hmac_key, 'base_id': prof.get_base_id(), 'parent_id': prof.get_id()}
        return trace_info

    def _start_with_trace(self, cnxt, trace, func, *args, **kwargs):
        if trace:
            profiler.init(**trace)
        if cnxt is not None:
            cnxt.update_store()
        return func(*args, **kwargs)

    def start(self, stack_id, func, *args, **kwargs):
        """Run the given method in a sub-thread."""
        if stack_id not in self.groups:
            self.groups[stack_id] = threadgroup.ThreadGroup()

        def log_exceptions(gt):
            try:
                gt.wait()
            except Exception:
                LOG.exception('Unhandled error in asynchronous task')
            except BaseException:
                pass
        req_cnxt = oslo_context.get_current()
        th = self.groups[stack_id].add_thread(self._start_with_trace, req_cnxt, self._serialize_profile_info(), func, *args, **kwargs)
        th.link(log_exceptions)
        return th

    def start_with_lock(self, cnxt, stack, engine_id, func, *args, **kwargs):
        """Run the method in sub-thread after acquiring the stack lock.

        Release the lock when the thread finishes.

        :param cnxt: RPC context
        :param stack: Stack to be operated on
        :type stack: heat.engine.parser.Stack
        :param engine_id: The UUID of the engine/worker acquiring the lock
        :param func: Callable to be invoked in sub-thread
        :type func: function or instancemethod
        :param args: Args to be passed to func
        :param kwargs: Keyword-args to be passed to func.
        """
        lock = stack_lock.StackLock(cnxt, stack.id, engine_id)
        with lock.thread_lock():
            th = self.start_with_acquired_lock(stack, lock, func, *args, **kwargs)
            return th

    def start_with_acquired_lock(self, stack, lock, func, *args, **kwargs):
        """Run the given method in a sub-thread with an existing stack lock.

        Release the provided lock when the thread finishes.

        :param stack: Stack to be operated on
        :type stack: heat.engine.parser.Stack
        :param lock: The acquired stack lock
        :type lock: heat.engine.stack_lock.StackLock
        :param func: Callable to be invoked in sub-thread
        :type func: function or instancemethod
        :param args: Args to be passed to func
        :param kwargs: Keyword-args to be passed to func

        """

        def _force_exit(*args):
            LOG.info('Graceful exit timeout exceeded, forcing exit.')
            os._exit(-1)

        def release(gt):
            """Callback function that will be passed to GreenThread.link().

            Persist the stack state to COMPLETE and FAILED close to
            releasing the lock to avoid race conditions.
            """
            if stack is not None and stack.defer_state_persist():
                stack.persist_state_and_release_lock(lock.engine_id)
                notify = kwargs.get('notify')
                if notify is not None and (not notify.signalled()):
                    notify.signal()
            else:
                try:
                    lock.release()
                except Exception:
                    signal.signal(signal.SIGALRM, _force_exit)
                    signal.alarm(5)
                    LOG.exception('FATAL. Failed stack_lock release. Exiting')
                    sys.exit(-1)
        stack.thread_group_mgr = self
        th = self.start(stack.id, func, *args, **kwargs)
        th.link(release)
        return th

    def add_timer(self, stack_id, func, *args, **kwargs):
        """Define a periodic task in the stack threadgroups.

        The task is run in a separate greenthread.

        Periodicity is cfg.CONF.periodic_interval
        """
        if stack_id not in self.groups:
            self.groups[stack_id] = threadgroup.ThreadGroup()
        self.groups[stack_id].add_timer(cfg.CONF.periodic_interval, func, None, *args, **kwargs)

    def add_msg_queue(self, stack_id, msg_queue):
        self.msg_queues[stack_id].append(msg_queue)

    def remove_msg_queue(self, gt, stack_id, msg_queue):
        for q in self.msg_queues.pop(stack_id, []):
            if q is not msg_queue:
                self.add_msg_queue(stack_id, q)

    def stop_timers(self, stack_id):
        if stack_id in self.groups:
            self.groups[stack_id].stop_timers()

    def stop(self, stack_id, graceful=False):
        """Stop any active threads on a stack."""
        if stack_id in self.groups:
            self.msg_queues.pop(stack_id, None)
            threadgroup = self.groups.pop(stack_id)
            threads = threadgroup.threads[:]
            threadgroup.stop(graceful)
            threadgroup.wait()
            links_done = dict(((th, False) for th in threads))

            def mark_done(gt, th):
                links_done[th] = True
            for th in threads:
                th.link(mark_done, th)
            while not all(links_done.values()):
                eventlet.sleep()

    def send(self, stack_id, message):
        for msg_queue in self.msg_queues.get(stack_id, []):
            msg_queue.put_nowait(message)