import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def undo_patch_thread_modules():
    for t in threading_modules_to_patch:
        try:
            t.start_new_thread = t._original_start_new_thread
        except:
            pass
        try:
            t.start_new = t._original_start_new_thread
        except:
            pass
        try:
            t._start_new_thread = t._original_start_new_thread
        except:
            pass