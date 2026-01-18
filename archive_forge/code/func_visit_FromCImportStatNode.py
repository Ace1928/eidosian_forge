from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def visit_FromCImportStatNode(self, node):
    module_name = node.module_name
    if module_name == u'cython.cimports' or module_name.startswith(u'cython.cimports.'):
        return self._create_cimport_from_import(node.pos, module_name, node.relative_level, node.imported_names)
    elif not node.relative_level and (module_name == u'cython' or module_name.startswith(u'cython.')):
        self._check_valid_cython_module(node.pos, module_name)
        submodule = (module_name + u'.')[7:]
        newimp = []
        for pos, name, as_name in node.imported_names:
            full_name = submodule + name
            qualified_name = u'cython.' + full_name
            if self.is_parallel_directive(qualified_name, node.pos):
                self.parallel_directives[as_name or name] = qualified_name
            elif self.is_cython_directive(full_name):
                self.directive_names[as_name or name] = full_name
            elif full_name in ['dataclasses', 'typing']:
                self.directive_names[as_name or name] = full_name
                newimp.append((pos, name, as_name))
            else:
                newimp.append((pos, name, as_name))
        if not newimp:
            return None
        node.imported_names = newimp
    return node