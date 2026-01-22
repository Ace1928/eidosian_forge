import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def _explicitly_disable_client_mode():
    global is_client_mode_enabled
    is_client_mode_enabled = False