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
def _dispatch_change_event(self, object, trait_name, old, new, handler):
    """ Prepare and dispatch a trait change event to a listener. """
    args = self.argument_transform(object, trait_name, old, new)
    try:
        self.dispatch(handler, *args)
    except Exception:
        handle_exception(object, trait_name, old, new)