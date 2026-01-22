import os
import sys
import warnings
from threading import RLock as threadRLock
from ctypes import CFUNCTYPE, c_int, CDLL, POINTER, c_uint
import numpy as np
import llvmlite.binding as ll
from llvmlite import ir
from numba.np.numpy_support import as_dtype
from numba.core import types, cgutils, config, errors
from numba.core.typing import signature
from numba.np.ufunc.wrappers import _wrapper_info
from numba.np.ufunc import ufuncbuilder
from numba.extending import overload, intrinsic
def _launch_threads():
    if not _backend_init_process_lock:
        _set_init_process_lock()
    with _backend_init_process_lock:
        with _backend_init_thread_lock:
            global _is_initialized
            if _is_initialized:
                return

            def select_known_backend(backend):
                """
                Loads a specific threading layer backend based on string
                """
                lib = None
                if backend.startswith('tbb'):
                    try:
                        _check_tbb_version_compatible()
                        from numba.np.ufunc import tbbpool as lib
                    except ImportError:
                        pass
                elif backend.startswith('omp'):
                    try:
                        from numba.np.ufunc import omppool as lib
                    except ImportError:
                        pass
                elif backend.startswith('workqueue'):
                    from numba.np.ufunc import workqueue as lib
                else:
                    msg = 'Unknown value specified for threading layer: %s'
                    raise ValueError(msg % backend)
                return lib

            def select_from_backends(backends):
                """
                Selects from presented backends and returns the first working
                """
                lib = None
                for backend in backends:
                    lib = select_known_backend(backend)
                    if lib is not None:
                        break
                else:
                    backend = ''
                return (lib, backend)
            t = str(config.THREADING_LAYER).lower()
            namedbackends = config.THREADING_LAYER_PRIORITY
            if not (len(namedbackends) == 3 and set(namedbackends) == {'tbb', 'omp', 'workqueue'}):
                raise ValueError("THREADING_LAYER_PRIORITY invalid: %s. It must be a permutation of {'tbb', 'omp', 'workqueue'}" % namedbackends)
            lib = None
            err_helpers = dict()
            err_helpers['TBB'] = 'Intel TBB is required, try:\n$ conda/pip install tbb'
            err_helpers['OSX_OMP'] = 'Intel OpenMP is required, try:\n$ conda/pip install intel-openmp'
            requirements = []

            def raise_with_hint(required):
                errmsg = 'No threading layer could be loaded.\n%s'
                hintmsg = 'HINT:\n%s'
                if len(required) == 0:
                    hint = ''
                if len(required) == 1:
                    hint = hintmsg % err_helpers[required[0]]
                if len(required) > 1:
                    options = '\nOR\n'.join([err_helpers[x] for x in required])
                    hint = hintmsg % ('One of:\n%s' % options)
                raise ValueError(errmsg % hint)
            if t in namedbackends:
                lib = select_known_backend(t)
                if not lib:
                    if t == 'tbb':
                        requirements.append('TBB')
                    elif t == 'omp' and _IS_OSX:
                        requirements.append('OSX_OMP')
                libname = t
            elif t in ['threadsafe', 'forksafe', 'safe']:
                available = ['tbb']
                requirements.append('TBB')
                if t == 'safe':
                    pass
                elif t == 'threadsafe':
                    if _IS_OSX:
                        requirements.append('OSX_OMP')
                    available.append('omp')
                elif t == 'forksafe':
                    if not _IS_LINUX:
                        available.append('omp')
                    if _IS_OSX:
                        requirements.append('OSX_OMP')
                    available.append('workqueue')
                else:
                    msg = 'No threading layer available for purpose %s'
                    raise ValueError(msg % t)
                lib, libname = select_from_backends(available)
            elif t == 'default':
                lib, libname = select_from_backends(namedbackends)
                if not lib:
                    requirements.append('TBB')
                    if _IS_OSX:
                        requirements.append('OSX_OMP')
            else:
                msg = "The threading layer requested '%s' is unknown to Numba."
                raise ValueError(msg % t)
            if not lib:
                raise_with_hint(requirements)
            ll.add_symbol('numba_parallel_for', lib.parallel_for)
            ll.add_symbol('do_scheduling_signed', lib.do_scheduling_signed)
            ll.add_symbol('do_scheduling_unsigned', lib.do_scheduling_unsigned)
            ll.add_symbol('allocate_sched', lib.allocate_sched)
            ll.add_symbol('deallocate_sched', lib.deallocate_sched)
            launch_threads = CFUNCTYPE(None, c_int)(lib.launch_threads)
            launch_threads(NUM_THREADS)
            _load_threading_functions(lib)
            global _threading_layer
            _threading_layer = libname
            _is_initialized = True