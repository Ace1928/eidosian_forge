import os
import sys
import itertools
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from typing import Dict, Iterator, List, Tuple
from pathlib import Path
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var
from distutils import log
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library
from distutils.sysconfig import _config_vars as _CONFIG_VARS  # noqa
def _write_stub_file(self, stub_file: str, ext: Extension, compile=False):
    log.info('writing stub loader for %s to %s', ext._full_name, stub_file)
    if compile and os.path.exists(stub_file):
        raise BaseError(stub_file + ' already exists! Please delete.')
    if not self.dry_run:
        f = open(stub_file, 'w')
        f.write('\n'.join(['def __bootstrap__():', '   global __bootstrap__, __file__, __loader__', '   import sys, os, pkg_resources, importlib.util' + if_dl(', dl'), '   __file__ = pkg_resources.resource_filename(__name__,%r)' % os.path.basename(ext._file_name), '   del __bootstrap__', "   if '__loader__' in globals():", '       del __loader__', if_dl('   old_flags = sys.getdlopenflags()'), '   old_dir = os.getcwd()', '   try:', '     os.chdir(os.path.dirname(__file__))', if_dl('     sys.setdlopenflags(dl.RTLD_NOW)'), '     spec = importlib.util.spec_from_file_location(', '                __name__, __file__)', '     mod = importlib.util.module_from_spec(spec)', '     spec.loader.exec_module(mod)', '   finally:', if_dl('     sys.setdlopenflags(old_flags)'), '     os.chdir(old_dir)', '__bootstrap__()', '']))
        f.close()
    if compile:
        self._compile_and_remove_stub(stub_file)