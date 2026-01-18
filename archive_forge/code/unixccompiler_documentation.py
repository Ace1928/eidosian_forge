import os
import sys
import subprocess
import shlex
from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log

    Build a static library in a separate sub-process.

    Parameters
    ----------
    objects : list or tuple of str
        List of paths to object files used to build the static library.
    output_libname : str
        The library name as an absolute or relative (if `output_dir` is used)
        path.
    output_dir : str, optional
        The path to the output directory. Default is None, in which case
        the ``output_dir`` attribute of the UnixCCompiler instance.
    debug : bool, optional
        This parameter is not used.
    target_lang : str, optional
        This parameter is not used.

    Returns
    -------
    None

    