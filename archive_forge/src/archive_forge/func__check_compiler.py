import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
def _check_compiler(self):
    old_config._check_compiler(self)
    from numpy.distutils.fcompiler import FCompiler, new_fcompiler
    if sys.platform == 'win32' and self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw'):
        if not self.compiler.initialized:
            try:
                self.compiler.initialize()
            except IOError as e:
                msg = textwrap.dedent('                        Could not initialize compiler instance: do you have Visual Studio\n                        installed?  If you are trying to build with MinGW, please use "python setup.py\n                        build -c mingw32" instead.  If you have Visual Studio installed, check it is\n                        correctly installed, and the right version (VS 2015 as of this writing).\n\n                        Original exception was: %s, and the Compiler class was %s\n                        ============================================================================') % (e, self.compiler.__class__.__name__)
                print(textwrap.dedent('                        ============================================================================'))
                raise distutils.errors.DistutilsPlatformError(msg) from e
        from distutils import msvc9compiler
        if msvc9compiler.get_build_version() >= 10:
            for ldflags in [self.compiler.ldflags_shared, self.compiler.ldflags_shared_debug]:
                if '/MANIFEST' not in ldflags:
                    ldflags.append('/MANIFEST')
    if not isinstance(self.fcompiler, FCompiler):
        self.fcompiler = new_fcompiler(compiler=self.fcompiler, dry_run=self.dry_run, force=1, c_compiler=self.compiler)
        if self.fcompiler is not None:
            self.fcompiler.customize(self.distribution)
            if self.fcompiler.get_version():
                self.fcompiler.customize_cmd(self)
                self.fcompiler.show_customization()