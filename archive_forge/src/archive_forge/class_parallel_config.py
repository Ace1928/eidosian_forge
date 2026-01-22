from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
class parallel_config:
    """Set the default backend or configuration for :class:`~joblib.Parallel`.

    This is an alternative to directly passing keyword arguments to the
    :class:`~joblib.Parallel` class constructor. It is particularly useful when
    calling into library code that uses joblib internally but does not expose
    the various parallel configuration arguments in its own API.

    Parameters
    ----------
    backend : str or ParallelBackendBase instance, default=None
        If ``backend`` is a string it must match a previously registered
        implementation using the :func:`~register_parallel_backend` function.

        By default the following backends are available:

        - 'loky': single-host, process-based parallelism (used by default),
        - 'threading': single-host, thread-based parallelism,
        - 'multiprocessing': legacy single-host, process-based parallelism.

        'loky' is recommended to run functions that manipulate Python objects.
        'threading' is a low-overhead alternative that is most efficient for
        functions that release the Global Interpreter Lock: e.g. I/O-bound
        code or CPU-bound code in a few calls to native code that explicitly
        releases the GIL. Note that on some rare systems (such as pyodide),
        multiprocessing and loky may not be available, in which case joblib
        defaults to threading.

        In addition, if the ``dask`` and ``distributed`` Python packages are
        installed, it is possible to use the 'dask' backend for better
        scheduling of nested parallel calls without over-subscription and
        potentially distribute parallel calls over a networked cluster of
        several hosts.

        It is also possible to use the distributed 'ray' backend for
        distributing the workload to a cluster of nodes. See more details
        in the Examples section below.

        Alternatively the backend can be passed directly as an instance.

    n_jobs : int, default=None
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when ``backend="loky"`` or the size of the
        thread-pool when ``backend="threading"``.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. For ``n_jobs`` below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for ``n_jobs=-2``, all
        CPUs but one are used.
        ``None`` is a marker for 'unset' that will be interpreted as
        ``n_jobs=1`` in most backends.

    verbose : int, default=0
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    temp_folder : str, default=None
        Folder to be used by the pool for memmapping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the ``JOBLIB_TEMP_FOLDER`` environment
          variable,
        - ``/dev/shm`` if the folder exists and is writable: this is a
          RAM disk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with ``TMP``, ``TMPDIR`` or ``TEMP`` environment
          variables, typically ``/tmp`` under Unix operating systems.

    max_nbytes int, str, or None, optional, default='1M'
        Threshold on the size of arrays passed to the workers that
        triggers automated memory mapping in temp_folder. Can be an int
        in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmapping of large arrays.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, default='r'
        Memmapping mode for numpy arrays passed to workers. None will
        disable memmapping, other modes defined in the numpy.memmap doc:
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
        Also, see 'max_nbytes' parameter documentation for more details.

    prefer: str in {'processes', 'threads'} or None, default=None
        Soft hint to choose the default backend.
        The default process-based backend is 'loky' and the default
        thread-based backend is 'threading'. Ignored if the ``backend``
        parameter is specified.

    require: 'sharedmem' or None, default=None
        Hard constraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based.

    inner_max_num_threads : int, default=None
        If not None, overwrites the limit set on the number of threads
        usable in some third-party library threadpools like OpenBLAS,
        MKL or OpenMP. This is only used with the ``loky`` backend.

    backend_params : dict
        Additional parameters to pass to the backend constructor when
        backend is a string.

    Notes
    -----
    Joblib tries to limit the oversubscription by limiting the number of
    threads usable in some third-party library threadpools like OpenBLAS, MKL
    or OpenMP. The default limit in each worker is set to
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be
    overwritten with the ``inner_max_num_threads`` argument which will be used
    to set this limit in the child processes.

    .. versionadded:: 1.3

    Examples
    --------
    >>> from operator import neg
    >>> with parallel_config(backend='threading'):
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    ...
    [-1, -2, -3, -4, -5]

    To use the 'ray' joblib backend add the following lines:

    >>> from ray.util.joblib import register_ray  # doctest: +SKIP
    >>> register_ray()  # doctest: +SKIP
    >>> with parallel_config(backend="ray"):  # doctest: +SKIP
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))
    [-1, -2, -3, -4, -5]

    """

    def __init__(self, backend=default_parallel_config['backend'], *, n_jobs=default_parallel_config['n_jobs'], verbose=default_parallel_config['verbose'], temp_folder=default_parallel_config['temp_folder'], max_nbytes=default_parallel_config['max_nbytes'], mmap_mode=default_parallel_config['mmap_mode'], prefer=default_parallel_config['prefer'], require=default_parallel_config['require'], inner_max_num_threads=None, **backend_params):
        self.old_parallel_config = getattr(_backend, 'config', default_parallel_config)
        backend = self._check_backend(backend, inner_max_num_threads, **backend_params)
        new_config = {'n_jobs': n_jobs, 'verbose': verbose, 'temp_folder': temp_folder, 'max_nbytes': max_nbytes, 'mmap_mode': mmap_mode, 'prefer': prefer, 'require': require, 'backend': backend}
        self.parallel_config = self.old_parallel_config.copy()
        self.parallel_config.update({k: v for k, v in new_config.items() if not isinstance(v, _Sentinel)})
        setattr(_backend, 'config', self.parallel_config)

    def _check_backend(self, backend, inner_max_num_threads, **backend_params):
        if backend is default_parallel_config['backend']:
            if inner_max_num_threads is not None or len(backend_params) > 0:
                raise ValueError('inner_max_num_threads and other constructor parameters backend_params are only supported when backend is not None.')
            return backend
        if isinstance(backend, str):
            if backend not in BACKENDS:
                if backend in EXTERNAL_BACKENDS:
                    register = EXTERNAL_BACKENDS[backend]
                    register()
                elif backend in MAYBE_AVAILABLE_BACKENDS:
                    warnings.warn(f"joblib backend '{backend}' is not available on your system, falling back to {DEFAULT_BACKEND}.", UserWarning, stacklevel=2)
                    BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]
                else:
                    raise ValueError(f'Invalid backend: {backend}, expected one of {sorted(BACKENDS.keys())}')
            backend = BACKENDS[backend](**backend_params)
        if inner_max_num_threads is not None:
            msg = f'{backend.__class__.__name__} does not accept setting the inner_max_num_threads argument.'
            assert backend.supports_inner_max_num_threads, msg
            backend.inner_max_num_threads = inner_max_num_threads
        if backend.nesting_level is None:
            parent_backend = self.old_parallel_config['backend']
            if parent_backend is default_parallel_config['backend']:
                nesting_level = 0
            else:
                nesting_level = parent_backend.nesting_level
            backend.nesting_level = nesting_level
        return backend

    def __enter__(self):
        return self.parallel_config

    def __exit__(self, type, value, traceback):
        self.unregister()

    def unregister(self):
        setattr(_backend, 'config', self.old_parallel_config)