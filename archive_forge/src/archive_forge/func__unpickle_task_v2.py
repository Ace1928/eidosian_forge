import inspect
from importlib import import_module
from celery._state import get_current_app
from celery.app.autoretry import add_autoretry_behaviour
from celery.exceptions import InvalidTaskError, NotRegistered
def _unpickle_task_v2(name, module=None):
    if module:
        import_module(module)
    return get_current_app().tasks[name]