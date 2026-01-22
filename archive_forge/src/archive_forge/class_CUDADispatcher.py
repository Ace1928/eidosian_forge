import numpy as np
import os
import sys
import ctypes
import functools
from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.caching import Cache, CacheImpl
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typing.typeof import Purpose, typeof
from numba.cuda.api import get_current_device
from numba.cuda.args import wrap_arg
from numba.cuda.compiler import compile_cuda, CUDACompiler
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
from numba.cuda import types as cuda_types
from numba import cuda
from numba import _dispatcher
from warnings import warn
class CUDADispatcher(Dispatcher, serialize.ReduceMixin):
    """
    CUDA Dispatcher object. When configured and called, the dispatcher will
    specialize itself for the given arguments (if no suitable specialized
    version already exists) & compute capability, and launch on the device
    associated with the current context.

    Dispatcher objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    """
    _fold_args = False
    targetdescr = cuda_target

    def __init__(self, py_func, targetoptions, pipeline_class=CUDACompiler):
        super().__init__(py_func, targetoptions=targetoptions, pipeline_class=pipeline_class)
        self._specialized = False
        self.specializations = {}

    @property
    def _numba_type_(self):
        return cuda_types.CUDADispatcher(self)

    def enable_caching(self):
        self._cache = CUDACache(self.py_func)

    @functools.lru_cache(maxsize=128)
    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        return _LaunchConfiguration(self, griddim, blockdim, stream, sharedmem)

    def __getitem__(self, args):
        if len(args) not in [2, 3, 4]:
            raise ValueError('must specify at least the griddim and blockdim')
        return self.configure(*args)

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        """Returns a 1D-configured dispatcher for a given number of tasks.

        This assumes that:

        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
          1-1 basis.
        - the kernel checks that the Global Thread ID is upper-bounded by
          ``ntasks``, and does nothing if it is not.

        :param ntasks: The number of tasks.
        :param tpb: The size of a block. An appropriate value is chosen if this
                    parameter is not supplied.
        :param stream: The stream on which the configured dispatcher will be
                       launched.
        :param sharedmem: The number of bytes of dynamic shared memory required
                          by the kernel.
        :return: A configured dispatcher, ready to launch on a set of
                 arguments."""
        return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)

    @property
    def extensions(self):
        """
        A list of objects that must have a `prepare_args` function. When a
        specialized kernel is called, each argument will be passed through
        to the `prepare_args` (from the last object in this list to the
        first). The arguments to `prepare_args` are:

        - `ty` the numba type of the argument
        - `val` the argument value itself
        - `stream` the CUDA stream used for the current call to the kernel
        - `retr` a list of zero-arg functions that you may want to append
          post-call cleanup work to.

        The `prepare_args` function must return a tuple `(ty, val)`, which
        will be passed in turn to the next right-most `extension`. After all
        the extensions have been called, the resulting `(ty, val)` will be
        passed into Numba's default argument marshalling logic.
        """
        return self.targetoptions.get('extensions')

    def __call__(self, *args, **kwargs):
        raise ValueError(missing_launch_config_msg)

    def call(self, args, griddim, blockdim, stream, sharedmem):
        """
        Compile if necessary and invoke this kernel with *args*.
        """
        if self.specialized:
            kernel = next(iter(self.overloads.values()))
        else:
            kernel = _dispatcher.Dispatcher._cuda_call(self, *args)
        kernel.launch(args, griddim, blockdim, stream, sharedmem)

    def _compile_for_args(self, *args, **kws):
        assert not kws
        argtypes = [self.typeof_pyval(a) for a in args]
        return self.compile(tuple(argtypes))

    def typeof_pyval(self, val):
        try:
            return typeof(val, Purpose.argument)
        except ValueError:
            if cuda.is_cuda_array(val):
                return typeof(cuda.as_cuda_array(val, sync=False), Purpose.argument)
            else:
                raise

    def specialize(self, *args):
        """
        Create a new instance of this dispatcher specialized for the given
        *args*.
        """
        cc = get_current_device().compute_capability
        argtypes = tuple([self.typingctx.resolve_argument_type(a) for a in args])
        if self.specialized:
            raise RuntimeError('Dispatcher already specialized')
        specialization = self.specializations.get((cc, argtypes))
        if specialization:
            return specialization
        targetoptions = self.targetoptions
        specialization = CUDADispatcher(self.py_func, targetoptions=targetoptions)
        specialization.compile(argtypes)
        specialization.disable_compile()
        specialization._specialized = True
        self.specializations[cc, argtypes] = specialization
        return specialization

    @property
    def specialized(self):
        """
        True if the Dispatcher has been specialized.
        """
        return self._specialized

    def get_regs_per_thread(self, signature=None):
        """
        Returns the number of registers used by each thread in this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get register
                          usage for. This may be omitted for a specialized
                          kernel.
        :return: The number of registers used by the compiled variant of the
                 kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].regs_per_thread
        if self.specialized:
            return next(iter(self.overloads.values())).regs_per_thread
        else:
            return {sig: overload.regs_per_thread for sig, overload in self.overloads.items()}

    def get_const_mem_size(self, signature=None):
        """
        Returns the size in bytes of constant memory used by this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get constant
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The size in bytes of constant memory allocated by the
                 compiled variant of the kernel for the given signature and
                 current device.
        """
        if signature is not None:
            return self.overloads[signature.args].const_mem_size
        if self.specialized:
            return next(iter(self.overloads.values())).const_mem_size
        else:
            return {sig: overload.const_mem_size for sig, overload in self.overloads.items()}

    def get_shared_mem_per_block(self, signature=None):
        """
        Returns the size in bytes of statically allocated shared memory
        for this kernel.

        :param signature: The signature of the compiled kernel to get shared
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of shared memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].shared_mem_per_block
        if self.specialized:
            return next(iter(self.overloads.values())).shared_mem_per_block
        else:
            return {sig: overload.shared_mem_per_block for sig, overload in self.overloads.items()}

    def get_max_threads_per_block(self, signature=None):
        """
        Returns the maximum allowable number of threads per block
        for this kernel. Exceeding this threshold will result in
        the kernel failing to launch.

        :param signature: The signature of the compiled kernel to get the max
                          threads per block for. This may be omitted for a
                          specialized kernel.
        :return: The maximum allowable threads per block for the compiled
                 variant of the kernel for the given signature and current
                 device.
        """
        if signature is not None:
            return self.overloads[signature.args].max_threads_per_block
        if self.specialized:
            return next(iter(self.overloads.values())).max_threads_per_block
        else:
            return {sig: overload.max_threads_per_block for sig, overload in self.overloads.items()}

    def get_local_mem_per_thread(self, signature=None):
        """
        Returns the size in bytes of local memory per thread
        for this kernel.

        :param signature: The signature of the compiled kernel to get local
                          memory usage for. This may be omitted for a
                          specialized kernel.
        :return: The amount of local memory allocated by the compiled variant
                 of the kernel for the given signature and current device.
        """
        if signature is not None:
            return self.overloads[signature.args].local_mem_per_thread
        if self.specialized:
            return next(iter(self.overloads.values())).local_mem_per_thread
        else:
            return {sig: overload.local_mem_per_thread for sig, overload in self.overloads.items()}

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows resolution of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        if self._can_compile:
            self.compile_device(tuple(args))
        func_name = self.py_func.__name__
        name = 'CallTemplate({0})'.format(func_name)
        call_template = typing.make_concrete_template(name, key=func_name, signatures=self.nopython_signatures)
        pysig = utils.pysignature(self.py_func)
        return (call_template, pysig, args, kws)

    def compile_device(self, args, return_type=None):
        """Compile the device function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
        if args not in self.overloads:
            with self._compiling_counter:
                debug = self.targetoptions.get('debug')
                lineinfo = self.targetoptions.get('lineinfo')
                inline = self.targetoptions.get('inline')
                fastmath = self.targetoptions.get('fastmath')
                nvvm_options = {'opt': 3 if self.targetoptions.get('opt') else 0, 'fastmath': fastmath}
                cc = get_current_device().compute_capability
                cres = compile_cuda(self.py_func, return_type, args, debug=debug, lineinfo=lineinfo, inline=inline, fastmath=fastmath, nvvm_options=nvvm_options, cc=cc)
                self.overloads[args] = cres
                cres.target_context.insert_user_function(cres.entry_point, cres.fndesc, [cres.library])
        else:
            cres = self.overloads[args]
        return cres

    def add_overload(self, kernel, argtypes):
        c_sig = [a._code for a in argtypes]
        self._insert(c_sig, kernel, cuda=True)
        self.overloads[argtypes] = kernel

    def compile(self, sig):
        """
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        """
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None or return_type == types.none
        if self.specialized:
            return next(iter(self.overloads.values()))
        else:
            kernel = self.overloads.get(argtypes)
            if kernel is not None:
                return kernel
        kernel = self._cache.load_overload(sig, self.targetctx)
        if kernel is not None:
            self._cache_hits[sig] += 1
        else:
            self._cache_misses[sig] += 1
            if not self._can_compile:
                raise RuntimeError('Compilation disabled')
            kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
            kernel.bind()
            self._cache.save_overload(sig, kernel)
        self.add_overload(kernel, argtypes)
        return kernel

    def inspect_llvm(self, signature=None):
        """
        Return the LLVM IR for this kernel.

        :param signature: A tuple of argument types.
        :return: The LLVM IR for the given signature, or a dict of LLVM IR
                 for all previously-encountered signatures.

        """
        device = self.targetoptions.get('device')
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_llvm_str()
            else:
                return self.overloads[signature].inspect_llvm()
        elif device:
            return {sig: overload.library.get_llvm_str() for sig, overload in self.overloads.items()}
        else:
            return {sig: overload.inspect_llvm() for sig, overload in self.overloads.items()}

    def inspect_asm(self, signature=None):
        """
        Return this kernel's PTX assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The PTX code for the given signature, or a dict of PTX codes
                 for all previously-encountered signatures.
        """
        cc = get_current_device().compute_capability
        device = self.targetoptions.get('device')
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_asm_str(cc)
            else:
                return self.overloads[signature].inspect_asm(cc)
        elif device:
            return {sig: overload.library.get_asm_str(cc) for sig, overload in self.overloads.items()}
        else:
            return {sig: overload.inspect_asm(cc) for sig, overload in self.overloads.items()}

    def inspect_sass_cfg(self, signature=None):
        """
        Return this kernel's CFG for the device in the current context.

        :param signature: A tuple of argument types.
        :return: The CFG for the given signature, or a dict of CFGs
                 for all previously-encountered signatures.

        The CFG for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
        if self.targetoptions.get('device'):
            raise RuntimeError('Cannot get the CFG of a device function')
        if signature is not None:
            return self.overloads[signature].inspect_sass_cfg()
        else:
            return {sig: defn.inspect_sass_cfg() for sig, defn in self.overloads.items()}

    def inspect_sass(self, signature=None):
        """
        Return this kernel's SASS assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The SASS code for the given signature, or a dict of SASS codes
                 for all previously-encountered signatures.

        SASS for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        """
        if self.targetoptions.get('device'):
            raise RuntimeError('Cannot inspect SASS of a device function')
        if signature is not None:
            return self.overloads[signature].inspect_sass()
        else:
            return {sig: defn.inspect_sass() for sig, defn in self.overloads.items()}

    def inspect_types(self, file=None):
        """
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        """
        if file is None:
            file = sys.stdout
        for _, defn in self.overloads.items():
            defn.inspect_types(file=file)

    @classmethod
    def _rebuild(cls, py_func, targetoptions):
        """
        Rebuild an instance.
        """
        instance = cls(py_func, targetoptions)
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
        return dict(py_func=self.py_func, targetoptions=self.targetoptions)