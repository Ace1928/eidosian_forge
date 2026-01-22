import json
import traceback
from contextvars import copy_context
from _plotly_utils.utils import PlotlyJSONEncoder
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.long_callback.managers import BaseLongCallbackManager
class CeleryManager(BaseLongCallbackManager):
    """Manage background execution of callbacks with a celery queue."""

    def __init__(self, celery_app, cache_by=None, expire=None):
        """
        Long callback manager that runs callback logic on a celery task queue,
        and stores results using a celery result backend.

        :param celery_app:
            A celery.Celery application instance that must be configured with a
            result backend. See the celery documentation for information on
            configuration options.
        :param cache_by:
            A list of zero-argument functions.  When provided, caching is enabled and
            the return values of these functions are combined with the callback
            function's input arguments and source code to generate cache keys.
        :param expire:
            If provided, a cache entry will be removed when it has not been accessed
            for ``expire`` seconds.  If not provided, the lifetime of cache entries
            is determined by the default behavior of the celery result backend.
        """
        try:
            import celery
            from celery.backends.base import DisabledBackend
        except ImportError as missing_imports:
            raise ImportError('CeleryLongCallbackManager requires extra dependencies which can be installed doing\n\n    $ pip install "dash[celery]"\n') from missing_imports
        if not isinstance(celery_app, celery.Celery):
            raise ValueError('First argument must be a celery.Celery object')
        if isinstance(celery_app.backend, DisabledBackend):
            raise ValueError('Celery instance must be configured with a result backend')
        self.handle = celery_app
        self.expire = expire
        super().__init__(cache_by)

    def terminate_job(self, job):
        if job is None:
            return
        self.handle.control.terminate(job)

    def terminate_unhealthy_job(self, job):
        task = self.get_task(job)
        if task and task.status in ('FAILURE', 'REVOKED'):
            return self.terminate_job(job)
        return False

    def job_running(self, job):
        future = self.get_task(job)
        return future and future.status in ('PENDING', 'RECEIVED', 'STARTED', 'RETRY', 'PROGRESS')

    def make_job_fn(self, fn, progress, key=None):
        return _make_job_fn(fn, self.handle, progress, key)

    def get_task(self, job):
        if job:
            return self.handle.AsyncResult(job)
        return None

    def clear_cache_entry(self, key):
        self.handle.backend.delete(key)

    def call_job_fn(self, key, job_fn, args, context):
        task = job_fn.delay(key, self._make_progress_key(key), args, context)
        return task.task_id

    def get_progress(self, key):
        progress_key = self._make_progress_key(key)
        progress_data = self.handle.backend.get(progress_key)
        if progress_data:
            self.handle.backend.delete(progress_key)
            return json.loads(progress_data)
        return None

    def result_ready(self, key):
        return self.handle.backend.get(key) is not None

    def get_result(self, key, job):
        result = self.handle.backend.get(key)
        if result is None:
            return self.UNDEFINED
        result = json.loads(result)
        if self.cache_by is None:
            self.clear_cache_entry(key)
        elif self.expire:
            self.handle.backend.expire(key, self.expire)
        self.clear_cache_entry(self._make_progress_key(key))
        self.terminate_job(job)
        return result