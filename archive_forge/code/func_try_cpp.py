import os, re
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.sysconfig import customize_compiler
from distutils import log
def try_cpp(self, body=None, headers=None, include_dirs=None, lang='c'):
    """Construct a source file from 'body' (a string containing lines
        of C/C++ code) and 'headers' (a list of header files to include)
        and run it through the preprocessor.  Return true if the
        preprocessor succeeded, false if there were any errors.
        ('body' probably isn't of much use, but what the heck.)
        """
    from distutils.ccompiler import CompileError
    self._check_compiler()
    ok = True
    try:
        self._preprocess(body, headers, include_dirs, lang)
    except CompileError:
        ok = False
    self._clean()
    return ok