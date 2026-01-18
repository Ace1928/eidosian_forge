import sys
import threading
import time
import traceback
from types import SimpleNamespace
def unregisterCallback(fn):
    """Unregister a previously registered callback.
    """
    callbacks.remove(fn)