import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
class _BaseLoadHandler(_BasePushHandler):

    def _handle(self):
        _BasePushHandler._handle(self)
        self.tokens = [_Token(self.i_line, self.instruction)]