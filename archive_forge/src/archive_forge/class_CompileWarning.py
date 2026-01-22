from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
class CompileWarning(PyrexWarning):

    def __init__(self, position=None, message=''):
        self.position = position
        Exception.__init__(self, format_position(position) + message)