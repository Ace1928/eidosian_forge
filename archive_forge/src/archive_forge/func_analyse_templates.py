from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def analyse_templates(self):
    if isinstance(self.base, CArrayDeclaratorNode):
        from .ExprNodes import TupleNode, NameNode
        template_node = self.base.dimension
        if isinstance(template_node, TupleNode):
            template_nodes = template_node.args
        elif isinstance(template_node, NameNode):
            template_nodes = [template_node]
        else:
            error(template_node.pos, 'Template arguments must be a list of names')
            return None
        self.templates = []
        for template in template_nodes:
            if isinstance(template, NameNode):
                self.templates.append(PyrexTypes.TemplatePlaceholderType(template.name))
            else:
                error(template.pos, 'Template arguments must be a list of names')
        self.base = self.base.base
        return self.templates
    else:
        return None