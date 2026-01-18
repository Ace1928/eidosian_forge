import os, re
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.sysconfig import customize_compiler
from distutils import log
def try_compile(self, body, headers=None, include_dirs=None, lang='c'):
    """Try to compile a source file built from 'body' and 'headers'.
        Return true on success, false otherwise.
        """
    from distutils.ccompiler import CompileError
    self._check_compiler()
    try:
        self._compile(body, headers, include_dirs, lang)
        ok = True
    except CompileError:
        ok = False
    log.info(ok and 'success!' or 'failure.')
    self._clean()
    return ok