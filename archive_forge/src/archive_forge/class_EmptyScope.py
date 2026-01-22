from __future__ import absolute_import
from .Nodes import *
from .ExprNodes import *
from .Errors import CompileError
class EmptyScope(object):

    def lookup(self, name):
        return None