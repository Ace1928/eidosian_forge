from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class MessageCollection(object):
    """Collect error/warnings messages first then sort"""

    def __init__(self):
        self.messages = set()

    def error(self, pos, message):
        self.messages.add((pos, True, message))

    def warning(self, pos, message):
        self.messages.add((pos, False, message))

    def report(self):
        for pos, is_error, message in sorted(self.messages):
            if is_error:
                error(pos, message)
            else:
                warning(pos, message, 2)