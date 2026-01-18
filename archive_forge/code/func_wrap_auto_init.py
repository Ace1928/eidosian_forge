import ray
import os
from functools import wraps
import threading
def wrap_auto_init(fn):

    @wraps(fn)
    def auto_init_wrapper(*args, **kwargs):
        auto_init_ray()
        return fn(*args, **kwargs)
    return auto_init_wrapper