import threading
from typing import TYPE_CHECKING, Any, Dict, Optional
from ray.train._internal import session
from ray.train._internal.storage import StorageContext
from ray.util.annotations import DeveloperAPI, PublicAPI
def _copy_doc(copy_func):

    def wrapped(func):
        func.__doc__ = copy_func.__doc__
        return func
    return wrapped