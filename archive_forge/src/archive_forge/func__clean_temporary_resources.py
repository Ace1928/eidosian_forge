from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util
from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError
from .numpy_pickle import dump, load, load_temporary_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
def _clean_temporary_resources(self, context_id=None, force=False, allow_non_empty=False):
    """Clean temporary resources created by a process-based pool"""
    if context_id is None:
        for context_id in list(self._cached_temp_folders):
            self._clean_temporary_resources(context_id, force=force, allow_non_empty=allow_non_empty)
    else:
        temp_folder = self._cached_temp_folders.get(context_id)
        if temp_folder and os.path.exists(temp_folder):
            for filename in os.listdir(temp_folder):
                if force:
                    resource_tracker.unregister(os.path.join(temp_folder, filename), 'file')
                else:
                    resource_tracker.maybe_unlink(os.path.join(temp_folder, filename), 'file')
            allow_non_empty |= force
            try:
                delete_folder(temp_folder, allow_non_empty=allow_non_empty)
                self._cached_temp_folders.pop(context_id, None)
                resource_tracker.unregister(temp_folder, 'folder')
                finalizer = self._finalizers.pop(context_id, None)
                if finalizer is not None:
                    atexit.unregister(finalizer)
            except OSError:
                pass