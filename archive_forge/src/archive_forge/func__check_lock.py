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
def _check_lock(self, handlers):
    """ Raises an exception if the specified handler stack is locked.
        """
    if handlers[-1].locked:
        raise TraitNotificationError('The traits notification exception handler is locked. No changes are allowed.')