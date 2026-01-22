from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
class PyDevIPCompleter6(IPCompleter):

    def __init__(self, *args, **kwargs):
        """ Create a Completer that reuses the advanced completion support of PyDev
            in addition to the completion support provided by IPython """
        IPCompleter.__init__(self, *args, **kwargs)

    @property
    def matchers(self):
        """All active matcher routines for completion"""
        return [self.file_matches, self.magic_matches, self.python_func_kw_matches, self.dict_key_matches]

    @matchers.setter
    def matchers(self, value):
        return