import sys
import time
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import partial
from weakref import WeakValueDictionary
from billiard.einfo import ExceptionInfo
from kombu.serialization import dumps, loads, prepare_accept_content
from kombu.serialization import registry as serializer_registry
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.url import maybe_sanitize_url
import celery.exceptions
from celery import current_app, group, maybe_signature, states
from celery._state import get_current_task
from celery.app.task import Context
from celery.exceptions import (BackendGetMetaError, BackendStoreError, ChordError, ImproperlyConfigured,
from celery.result import GroupResult, ResultBase, ResultSet, allow_join_result, result_from_tuple
from celery.utils.collections import BufferMap
from celery.utils.functional import LRUCache, arity_greater
from celery.utils.log import get_logger
from celery.utils.serialization import (create_exception_cls, ensure_serializable, get_pickleable_exception,
from celery.utils.time import get_exponential_backoff_interval
class BaseKeyValueStoreBackend(Backend):
    key_t = ensure_bytes
    task_keyprefix = 'celery-task-meta-'
    group_keyprefix = 'celery-taskset-meta-'
    chord_keyprefix = 'chord-unlock-'
    implements_incr = False

    def __init__(self, *args, **kwargs):
        if hasattr(self.key_t, '__func__'):
            self.key_t = self.key_t.__func__
        super().__init__(*args, **kwargs)
        self._add_global_keyprefix()
        self._encode_prefixes()
        if self.implements_incr:
            self.apply_chord = self._apply_chord_incr

    def _add_global_keyprefix(self):
        """
        This method prepends the global keyprefix to the existing keyprefixes.

        This method checks if a global keyprefix is configured in `result_backend_transport_options` using the
        `global_keyprefix` key. If so, then it is prepended to the task, group and chord key prefixes.
        """
        global_keyprefix = self.app.conf.get('result_backend_transport_options', {}).get('global_keyprefix', None)
        if global_keyprefix:
            self.task_keyprefix = f'{global_keyprefix}_{self.task_keyprefix}'
            self.group_keyprefix = f'{global_keyprefix}_{self.group_keyprefix}'
            self.chord_keyprefix = f'{global_keyprefix}_{self.chord_keyprefix}'

    def _encode_prefixes(self):
        self.task_keyprefix = self.key_t(self.task_keyprefix)
        self.group_keyprefix = self.key_t(self.group_keyprefix)
        self.chord_keyprefix = self.key_t(self.chord_keyprefix)

    def get(self, key):
        raise NotImplementedError('Must implement the get method.')

    def mget(self, keys):
        raise NotImplementedError('Does not support get_many')

    def _set_with_state(self, key, value, state):
        return self.set(key, value)

    def set(self, key, value):
        raise NotImplementedError('Must implement the set method.')

    def delete(self, key):
        raise NotImplementedError('Must implement the delete method')

    def incr(self, key):
        raise NotImplementedError('Does not implement incr')

    def expire(self, key, value):
        pass

    def get_key_for_task(self, task_id, key=''):
        """Get the cache key for a task by id."""
        if not task_id:
            raise ValueError(f'task_id must not be empty. Got {task_id} instead.')
        return self._get_key_for(self.task_keyprefix, task_id, key)

    def get_key_for_group(self, group_id, key=''):
        """Get the cache key for a group by id."""
        if not group_id:
            raise ValueError(f'group_id must not be empty. Got {group_id} instead.')
        return self._get_key_for(self.group_keyprefix, group_id, key)

    def get_key_for_chord(self, group_id, key=''):
        """Get the cache key for the chord waiting on group with given id."""
        if not group_id:
            raise ValueError(f'group_id must not be empty. Got {group_id} instead.')
        return self._get_key_for(self.chord_keyprefix, group_id, key)

    def _get_key_for(self, prefix, id, key=''):
        key_t = self.key_t
        return key_t('').join([prefix, key_t(id), key_t(key)])

    def _strip_prefix(self, key):
        """Take bytes: emit string."""
        key = self.key_t(key)
        for prefix in (self.task_keyprefix, self.group_keyprefix):
            if key.startswith(prefix):
                return bytes_to_str(key[len(prefix):])
        return bytes_to_str(key)

    def _filter_ready(self, values, READY_STATES=states.READY_STATES):
        for k, value in values:
            if value is not None:
                value = self.decode_result(value)
                if value['status'] in READY_STATES:
                    yield (k, value)

    def _mget_to_results(self, values, keys, READY_STATES=states.READY_STATES):
        if hasattr(values, 'items'):
            return {self._strip_prefix(k): v for k, v in self._filter_ready(values.items(), READY_STATES)}
        else:
            return {bytes_to_str(keys[i]): v for i, v in self._filter_ready(enumerate(values), READY_STATES)}

    def get_many(self, task_ids, timeout=None, interval=0.5, no_ack=True, on_message=None, on_interval=None, max_iterations=None, READY_STATES=states.READY_STATES):
        interval = 0.5 if interval is None else interval
        ids = task_ids if isinstance(task_ids, set) else set(task_ids)
        cached_ids = set()
        cache = self._cache
        for task_id in ids:
            try:
                cached = cache[task_id]
            except KeyError:
                pass
            else:
                if cached['status'] in READY_STATES:
                    yield (bytes_to_str(task_id), cached)
                    cached_ids.add(task_id)
        ids.difference_update(cached_ids)
        iterations = 0
        while ids:
            keys = list(ids)
            r = self._mget_to_results(self.mget([self.get_key_for_task(k) for k in keys]), keys, READY_STATES)
            cache.update(r)
            ids.difference_update({bytes_to_str(v) for v in r})
            for key, value in r.items():
                if on_message is not None:
                    on_message(value)
                yield (bytes_to_str(key), value)
            if timeout and iterations * interval >= timeout:
                raise TimeoutError(f'Operation timed out ({timeout})')
            if on_interval:
                on_interval()
            time.sleep(interval)
            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break

    def _forget(self, task_id):
        self.delete(self.get_key_for_task(task_id))

    def _store_result(self, task_id, result, state, traceback=None, request=None, **kwargs):
        meta = self._get_result_meta(result=result, state=state, traceback=traceback, request=request)
        meta['task_id'] = bytes_to_str(task_id)
        current_meta = self._get_task_meta_for(task_id)
        if current_meta['status'] == states.SUCCESS:
            return result
        try:
            self._set_with_state(self.get_key_for_task(task_id), self.encode(meta), state)
        except BackendStoreError as ex:
            raise BackendStoreError(str(ex), state=state, task_id=task_id) from ex
        return result

    def _save_group(self, group_id, result):
        self._set_with_state(self.get_key_for_group(group_id), self.encode({'result': result.as_tuple()}), states.SUCCESS)
        return result

    def _delete_group(self, group_id):
        self.delete(self.get_key_for_group(group_id))

    def _get_task_meta_for(self, task_id):
        """Get task meta-data for a task by id."""
        meta = self.get(self.get_key_for_task(task_id))
        if not meta:
            return {'status': states.PENDING, 'result': None}
        return self.decode_result(meta)

    def _restore_group(self, group_id):
        """Get task meta-data for a task by id."""
        meta = self.get(self.get_key_for_group(group_id))
        if meta:
            meta = self.decode(meta)
            result = meta['result']
            meta['result'] = result_from_tuple(result, self.app)
            return meta

    def _apply_chord_incr(self, header_result_args, body, **kwargs):
        self.ensure_chords_allowed()
        header_result = self.app.GroupResult(*header_result_args)
        header_result.save(backend=self)

    def on_chord_part_return(self, request, state, result, **kwargs):
        if not self.implements_incr:
            return
        app = self.app
        gid = request.group
        if not gid:
            return
        key = self.get_key_for_chord(gid)
        try:
            deps = GroupResult.restore(gid, backend=self)
        except Exception as exc:
            callback = maybe_signature(request.chord, app=app)
            logger.exception('Chord %r raised: %r', gid, exc)
            return self.chord_error_from_stack(callback, ChordError(f'Cannot restore group: {exc!r}'))
        if deps is None:
            try:
                raise ValueError(gid)
            except ValueError as exc:
                callback = maybe_signature(request.chord, app=app)
                logger.exception('Chord callback %r raised: %r', gid, exc)
                return self.chord_error_from_stack(callback, ChordError(f'GroupResult {gid} no longer exists'))
        val = self.incr(key)
        size = request.chord.get('chord_size')
        if size is None:
            size = len(deps)
        if val > size:
            logger.warning('Chord counter incremented too many times for %r', gid)
        elif val == size:
            callback = maybe_signature(request.chord, app=app)
            j = deps.join_native if deps.supports_native_join else deps.join
            try:
                with allow_join_result():
                    ret = j(timeout=app.conf.result_chord_join_timeout, propagate=True)
            except Exception as exc:
                try:
                    culprit = next(deps._failed_join_report())
                    reason = 'Dependency {0.id} raised {1!r}'.format(culprit, exc)
                except StopIteration:
                    reason = repr(exc)
                logger.exception('Chord %r raised: %r', gid, reason)
                self.chord_error_from_stack(callback, ChordError(reason))
            else:
                try:
                    callback.delay(ret)
                except Exception as exc:
                    logger.exception('Chord %r raised: %r', gid, exc)
                    self.chord_error_from_stack(callback, ChordError(f'Callback error: {exc!r}'))
            finally:
                deps.delete()
                self.delete(key)
        else:
            self.expire(key, self.expires)