import logging
from contextlib import contextmanager
from vine.utils import wraps
from celery import states
from celery.backends.base import BaseBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.time import maybe_timedelta
from .models import Task, TaskExtended, TaskSet
from .session import SessionManager
@retry
def _get_task_meta_for(self, task_id):
    """Get task meta-data for a task by id."""
    session = self.ResultSession()
    with session_cleanup(session):
        task = list(session.query(self.task_cls).filter(self.task_cls.task_id == task_id))
        task = task and task[0]
        if not task:
            task = self.task_cls(task_id)
            task.status = states.PENDING
            task.result = None
        data = task.to_dict()
        if data.get('args', None) is not None:
            data['args'] = self.decode(data['args'])
        if data.get('kwargs', None) is not None:
            data['kwargs'] = self.decode(data['kwargs'])
        return self.meta_from_decoded(data)