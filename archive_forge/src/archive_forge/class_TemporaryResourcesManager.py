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
class TemporaryResourcesManager(object):
    """Stateful object able to manage temporary folder and pickles

    It exposes:
    - a per-context folder name resolving API that memmap-based reducers will
      rely on to know where to pickle the temporary memmaps
    - a temporary file/folder management API that internally uses the
      resource_tracker.
    """

    def __init__(self, temp_folder_root=None, context_id=None):
        self._current_temp_folder = None
        self._temp_folder_root = temp_folder_root
        self._use_shared_mem = None
        self._cached_temp_folders = dict()
        self._id = uuid4().hex
        self._finalizers = {}
        if context_id is None:
            context_id = uuid4().hex
        self.set_current_context(context_id)

    def set_current_context(self, context_id):
        self._current_context_id = context_id
        self.register_new_context(context_id)

    def register_new_context(self, context_id):
        if context_id in self._cached_temp_folders:
            return
        else:
            new_folder_name = 'joblib_memmapping_folder_{}_{}_{}'.format(os.getpid(), self._id, context_id)
            new_folder_path, _ = _get_temp_dir(new_folder_name, self._temp_folder_root)
            self.register_folder_finalizer(new_folder_path, context_id)
            self._cached_temp_folders[context_id] = new_folder_path

    def resolve_temp_folder_name(self):
        """Return a folder name specific to the currently activated context"""
        return self._cached_temp_folders[self._current_context_id]

    def register_folder_finalizer(self, pool_subfolder, context_id):
        pool_module_name = whichmodule(delete_folder, 'delete_folder')
        resource_tracker.register(pool_subfolder, 'folder')

        def _cleanup():
            delete_folder = __import__(pool_module_name, fromlist=['delete_folder']).delete_folder
            try:
                delete_folder(pool_subfolder, allow_non_empty=True)
                resource_tracker.unregister(pool_subfolder, 'folder')
            except OSError:
                warnings.warn('Failed to delete temporary folder: {}'.format(pool_subfolder))
        self._finalizers[context_id] = atexit.register(_cleanup)

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