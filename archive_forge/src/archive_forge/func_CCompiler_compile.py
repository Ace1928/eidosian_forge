import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def CCompiler_compile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    """
    Compile one or more source files.

    Please refer to the Python distutils API reference for more details.

    Parameters
    ----------
    sources : list of str
        A list of filenames
    output_dir : str, optional
        Path to the output directory.
    macros : list of tuples
        A list of macro definitions.
    include_dirs : list of str, optional
        The directories to add to the default include file search path for
        this compilation only.
    debug : bool, optional
        Whether or not to output debug symbols in or alongside the object
        file(s).
    extra_preargs, extra_postargs : ?
        Extra pre- and post-arguments.
    depends : list of str, optional
        A list of file names that all targets depend on.

    Returns
    -------
    objects : list of str
        A list of object file names, one per source file `sources`.

    Raises
    ------
    CompileError
        If compilation fails.

    """
    global _job_semaphore
    jobs = get_num_build_jobs()
    with _global_lock:
        if _job_semaphore is None:
            _job_semaphore = threading.Semaphore(jobs)
    if not sources:
        return []
    from numpy.distutils.fcompiler import FCompiler, FORTRAN_COMMON_FIXED_EXTENSIONS, has_f90_header
    if isinstance(self, FCompiler):
        display = []
        for fc in ['f77', 'f90', 'fix']:
            fcomp = getattr(self, 'compiler_' + fc)
            if fcomp is None:
                continue
            display.append('Fortran %s compiler: %s' % (fc, ' '.join(fcomp)))
        display = '\n'.join(display)
    else:
        ccomp = self.compiler_so
        display = 'C compiler: %s\n' % (' '.join(ccomp),)
    log.info(display)
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    display = "compile options: '%s'" % ' '.join(cc_args)
    if extra_postargs:
        display += "\nextra options: '%s'" % ' '.join(extra_postargs)
    log.info(display)

    def single_compile(args):
        obj, (src, ext) = args
        if not _needs_build(obj, cc_args, extra_postargs, pp_opts):
            return
        while True:
            with _global_lock:
                if obj not in _processing_files:
                    _processing_files.add(obj)
                    break
            time.sleep(0.1)
        try:
            with _job_semaphore:
                self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        finally:
            with _global_lock:
                _processing_files.remove(obj)
    if isinstance(self, FCompiler):
        objects_to_build = list(build.keys())
        f77_objects, other_objects = ([], [])
        for obj in objects:
            if obj in objects_to_build:
                src, ext = build[obj]
                if self.compiler_type == 'absoft':
                    obj = cyg2win32(obj)
                    src = cyg2win32(src)
                if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS and (not has_f90_header(src)):
                    f77_objects.append((obj, (src, ext)))
                else:
                    other_objects.append((obj, (src, ext)))
        build_items = f77_objects
        for o in other_objects:
            single_compile(o)
    else:
        build_items = build.items()
    if len(build) > 1 and jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as pool:
            res = pool.map(single_compile, build_items)
        list(res)
    else:
        for o in build_items:
            single_compile(o)
    return objects