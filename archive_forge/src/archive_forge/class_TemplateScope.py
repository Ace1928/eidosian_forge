from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class TemplateScope(Scope):

    def __init__(self, name, outer_scope):
        Scope.__init__(self, name, outer_scope, None)
        self.directives = outer_scope.directives