import os
import sys
import re
import shlex
import itertools
from . import sysconfig
from ._modified import newer
from .ccompiler import CCompiler, gen_preprocess_options, gen_lib_options
from .errors import DistutilsExecError, CompileError, LibError, LinkError
from ._log import log
from ._macos_compat import compiler_fixup
def _split_aix(cmd):
    """
    AIX platforms prefix the compiler with the ld_so_aix
    script, so split that from the linker command.

    >>> _split_aix(['a', 'b', 'c'])
    ([], ['a', 'b', 'c'])
    >>> _split_aix(['/bin/foo/ld_so_aix', 'gcc'])
    (['/bin/foo/ld_so_aix'], ['gcc'])
    """
    pivot = os.path.basename(cmd[0]) == 'ld_so_aix'
    return (cmd[:pivot], cmd[pivot:])