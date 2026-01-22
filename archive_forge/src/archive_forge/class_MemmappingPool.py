import copyreg
import sys
import warnings
from time import sleep
from pickle import Pickler
from pickle import HIGHEST_PROTOCOL
from io import BytesIO
from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from ._multiprocessing_helpers import mp, assert_spawning
from multiprocessing.pool import Pool
class MemmappingPool(PicklingPool):
    """Process pool that shares large arrays to avoid memory copy.

    This drop-in replacement for `multiprocessing.pool.Pool` makes
    it possible to work efficiently with shared memory in a numpy
    context.

    Existing instances of numpy.memmap are preserved: the child
    suprocesses will have access to the same shared memory in the
    original mode except for the 'w+' mode that is automatically
    transformed as 'r+' to avoid zeroing the original data upon
    instantiation.

    Furthermore large arrays from the parent process are automatically
    dumped to a temporary folder on the filesystem such as child
    processes to access their content via memmapping (file system
    backed shared memory).

    Note: it is important to call the terminate method to collect
    the temporary folder used by the pool.

    Parameters
    ----------
    processes: int, optional
        Number of worker processes running concurrently in the pool.
    initializer: callable, optional
        Callable executed on worker process creation.
    initargs: tuple, optional
        Arguments passed to the initializer callable.
    temp_folder: (str, callable) optional
        If str:
          Folder to be used by the pool for memmapping large arrays
          for sharing memory with worker processes. If None, this will try in
          order:
          - a folder pointed by the JOBLIB_TEMP_FOLDER environment variable,
          - /dev/shm if the folder exists and is writable: this is a RAMdisk
            filesystem available by default on modern Linux distributions,
          - the default system temporary folder that can be overridden
            with TMP, TMPDIR or TEMP environment variables, typically /tmp
            under Unix operating systems.
        if callable:
            An callable in charge of dynamically resolving a temporary folder
            for memmapping large arrays.
    max_nbytes int or None, optional, 1e6 by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memory mapping in temp_folder.
        Use None to disable memmapping of large arrays.
    mmap_mode: {'r+', 'r', 'w+', 'c'}
        Memmapping mode for numpy arrays passed to workers.
        See 'max_nbytes' parameter documentation for more details.
    forward_reducers: dictionary, optional
        Reducers used to pickle objects passed from master to worker
        processes: see below.
    backward_reducers: dictionary, optional
        Reducers used to pickle return values from workers back to the
        master process.
    verbose: int, optional
        Make it possible to monitor how the communication of numpy arrays
        with the subprocess is handled (pickling or memmapping)
    prewarm: bool or str, optional, "auto" by default.
        If True, force a read on newly memmapped array to make sure that OS
        pre-cache it in memory. This can be useful to avoid concurrent disk
        access when the same data array is passed to different worker
        processes. If "auto" (by default), prewarm is set to True, unless the
        Linux shared memory partition /dev/shm is available and used as temp
        folder.

    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that give an instance of `type` will return
    a tuple `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for more
    details.

    """

    def __init__(self, processes=None, temp_folder=None, max_nbytes=1000000.0, mmap_mode='r', forward_reducers=None, backward_reducers=None, verbose=0, context_id=None, prewarm=False, **kwargs):
        if context_id is not None:
            warnings.warn('context_id is deprecated and ignored in joblib 0.9.4 and will be removed in 0.11', DeprecationWarning)
        manager = TemporaryResourcesManager(temp_folder)
        self._temp_folder_manager = manager
        forward_reducers, backward_reducers = get_memmapping_reducers(temp_folder_resolver=manager.resolve_temp_folder_name, max_nbytes=max_nbytes, mmap_mode=mmap_mode, forward_reducers=forward_reducers, backward_reducers=backward_reducers, verbose=verbose, unlink_on_gc_collect=False, prewarm=prewarm)
        poolargs = dict(processes=processes, forward_reducers=forward_reducers, backward_reducers=backward_reducers)
        poolargs.update(kwargs)
        super(MemmappingPool, self).__init__(**poolargs)

    def terminate(self):
        n_retries = 10
        for i in range(n_retries):
            try:
                super(MemmappingPool, self).terminate()
                break
            except OSError as e:
                if isinstance(e, WindowsError):
                    sleep(0.1)
                    if i + 1 == n_retries:
                        warnings.warn('Failed to terminate worker processes in multiprocessing pool: %r' % e)
        self._temp_folder_manager._clean_temporary_resources()

    @property
    def _temp_folder(self):
        if getattr(self, '_cached_temp_folder', None) is not None:
            return self._cached_temp_folder
        else:
            self._cached_temp_folder = self._temp_folder_manager.resolve_temp_folder_name()
            return self._cached_temp_folder