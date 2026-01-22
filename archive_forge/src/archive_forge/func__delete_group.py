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
def _delete_group(self, group_id):
    """Delete meta-data for group by id."""
    session = self.ResultSession()
    with session_cleanup(session):
        session.query(self.taskset_cls).filter(self.taskset_cls.taskset_id == group_id).delete()
        session.flush()
        session.commit()