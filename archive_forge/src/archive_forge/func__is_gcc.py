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
def _is_gcc(self):
    cc_var = sysconfig.get_config_var('CC')
    compiler = os.path.basename(shlex.split(cc_var)[0])
    return 'gcc' in compiler or 'g++' in compiler