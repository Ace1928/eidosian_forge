from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CompleteFiles(Completer):
    """
    Completes file names based on a glob pattern
    """

    def __init__(self, globPattern='*', **kw):
        Completer.__init__(self, **kw)
        self._globPattern = globPattern

    def _description(self, optName):
        if self._descr is not None:
            return f'{self._descr} ({self._globPattern})'
        else:
            return f'{optName} ({self._globPattern})'

    def _shellCode(self, optName, shellType):
        if shellType == _ZSH:
            return '{}:{}:_files -g "{}"'.format(self._repeatFlag, self._description(optName), self._globPattern)
        raise NotImplementedError(f'Unknown shellType {shellType!r}')