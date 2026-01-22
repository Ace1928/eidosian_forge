import contextlib
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import tempfile
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import (
import distutils.ccompiler
import distutils.errors
class ParallelCompile:
    """
    Make a parallel compile function. Inspired by
    numpy.distutils.ccompiler.CCompiler.compile and cppimport.

    This takes several arguments that allow you to customize the compile
    function created:

    envvar:
        Set an environment variable to control the compilation threads, like
        NPY_NUM_BUILD_JOBS
    default:
        0 will automatically multithread, or 1 will only multithread if the
        envvar is set.
    max:
        The limit for automatic multithreading if non-zero
    needs_recompile:
        A function of (obj, src) that returns True when recompile is needed.  No
        effect in isolated mode; use ccache instead, see
        https://github.com/matplotlib/matplotlib/issues/1507/

    To use::

        ParallelCompile("NPY_NUM_BUILD_JOBS").install()

    or::

        with ParallelCompile("NPY_NUM_BUILD_JOBS"):
            setup(...)

    By default, this assumes all files need to be recompiled. A smarter
    function can be provided via needs_recompile.  If the output has not yet
    been generated, the compile will always run, and this function is not
    called.
    """
    __slots__ = ('envvar', 'default', 'max', '_old', 'needs_recompile')

    def __init__(self, envvar: Optional[str]=None, default: int=0, max: int=0, needs_recompile: Callable[[str, str], bool]=no_recompile) -> None:
        self.envvar = envvar
        self.default = default
        self.max = max
        self.needs_recompile = needs_recompile
        self._old: List[CCompilerMethod] = []

    def function(self) -> CCompilerMethod:
        """
        Builds a function object usable as distutils.ccompiler.CCompiler.compile.
        """

        def compile_function(compiler: distutils.ccompiler.CCompiler, sources: List[str], output_dir: Optional[str]=None, macros: Optional[Union[Tuple[str], Tuple[str, Optional[str]]]]=None, include_dirs: Optional[List[str]]=None, debug: bool=False, extra_preargs: Optional[List[str]]=None, extra_postargs: Optional[List[str]]=None, depends: Optional[List[str]]=None) -> Any:
            macros, objects, extra_postargs, pp_opts, build = compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            cc_args = compiler._get_cc_args(pp_opts, debug, extra_preargs)
            threads = self.default
            if self.envvar is not None:
                threads = int(os.environ.get(self.envvar, self.default))

            def _single_compile(obj: Any) -> None:
                try:
                    src, ext = build[obj]
                except KeyError:
                    return
                if not os.path.exists(obj) or self.needs_recompile(obj, src):
                    compiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
            try:
                import multiprocessing.synchronize
                from multiprocessing.pool import ThreadPool
            except ImportError:
                threads = 1
            if threads == 0:
                try:
                    threads = multiprocessing.cpu_count()
                    threads = self.max if self.max and self.max < threads else threads
                except NotImplementedError:
                    threads = 1
            if threads > 1:
                with ThreadPool(threads) as pool:
                    for _ in pool.imap_unordered(_single_compile, objects):
                        pass
            else:
                for ob in objects:
                    _single_compile(ob)
            return objects
        return compile_function

    def install(self: S) -> S:
        """
        Installs the compile function into distutils.ccompiler.CCompiler.compile.
        """
        distutils.ccompiler.CCompiler.compile = self.function()
        return self

    def __enter__(self: S) -> S:
        self._old.append(distutils.ccompiler.CCompiler.compile)
        return self.install()

    def __exit__(self, *args: Any) -> None:
        distutils.ccompiler.CCompiler.compile = self._old.pop()