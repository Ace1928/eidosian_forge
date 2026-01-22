import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
@contextlib.contextmanager
def _filter_build_errors(self, ext):
    try:
        yield
    except (CCompilerError, DistutilsError, CompileError) as e:
        if not ext.optional:
            raise
        self.warn('building extension "%s" failed: %s' % (ext.name, e))