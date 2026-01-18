import contextlib
import logging
import threading
from threading import local as thread_local
from threading import Thread
import traceback
from types import MethodType
import weakref
import sys
from .constants import ComparisonMode, TraitKind
from .trait_base import Uninitialized
from .trait_errors import TraitNotificationError
def ui_dispatch(handler, *args, **kw):
    if threading.current_thread().ident == ui_thread:
        handler(*args, **kw)
    else:
        ui_handler(handler, *args, **kw)