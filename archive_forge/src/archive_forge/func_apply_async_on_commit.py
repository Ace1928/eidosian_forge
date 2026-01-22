import functools
from django.db import transaction
from celery.app.task import Task
def apply_async_on_commit(self, *args, **kwargs):
    """Call :meth:`~celery.app.task.Task.apply_async` with Django's ``on_commit()``."""
    return transaction.on_commit(functools.partial(self.apply_async, *args, **kwargs))