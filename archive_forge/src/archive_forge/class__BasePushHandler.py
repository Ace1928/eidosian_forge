import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
class _BasePushHandler(_BaseHandler):

    def _handle(self):
        self.stack.push(self)