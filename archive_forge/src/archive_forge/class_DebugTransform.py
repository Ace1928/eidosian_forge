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
class DebugTransform(CythonTransform):
    """
    Write debug information for this Cython module.
    """

    def __init__(self, context, options, result):
        super(DebugTransform, self).__init__(context)
        self.visited = set()
        self.tb = self.context.gdb_debug_outputwriter
        self.c_output_file = result.c_file
        self.nested_funcdefs = []
        self.register_stepinto = False

    def visit_ModuleNode(self, node):
        self.tb.module_name = node.full_module_name
        attrs = dict(module_name=node.full_module_name, filename=node.pos[0].filename, c_filename=self.c_output_file)
        self.tb.start('Module', attrs)
        self.tb.start('Functions')
        self.visitchildren(node)
        for nested_funcdef in self.nested_funcdefs:
            self.visit_FuncDefNode(nested_funcdef)
        self.register_stepinto = True
        self.serialize_modulenode_as_function(node)
        self.register_stepinto = False
        self.tb.end('Functions')
        self.tb.start('Globals')
        entries = {}
        for k, v in node.scope.entries.items():
            if v.qualified_name not in self.visited and (not v.name.startswith('__pyx_')) and (not v.type.is_cfunction) and (not v.type.is_extension_type):
                entries[k] = v
        self.serialize_local_variables(entries)
        self.tb.end('Globals')
        return node

    def visit_FuncDefNode(self, node):
        self.visited.add(node.local_scope.qualified_name)
        if getattr(node, 'is_wrapper', False):
            return node
        if self.register_stepinto:
            self.nested_funcdefs.append(node)
            return node
        if node.py_func is None:
            pf_cname = ''
        else:
            pf_cname = node.py_func.entry.func_cname
        cname = node.entry.pyfunc_cname or node.entry.func_cname
        attrs = dict(name=node.entry.name or getattr(node, 'name', '<unknown>'), cname=cname, pf_cname=pf_cname, qualified_name=node.local_scope.qualified_name, lineno=str(node.pos[1]))
        self.tb.start('Function', attrs=attrs)
        self.tb.start('Locals')
        self.serialize_local_variables(node.local_scope.entries)
        self.tb.end('Locals')
        self.tb.start('Arguments')
        for arg in node.local_scope.arg_entries:
            self.tb.start(arg.name)
            self.tb.end(arg.name)
        self.tb.end('Arguments')
        self.tb.start('StepIntoFunctions')
        self.register_stepinto = True
        self.visitchildren(node)
        self.register_stepinto = False
        self.tb.end('StepIntoFunctions')
        self.tb.end('Function')
        return node

    def visit_NameNode(self, node):
        if self.register_stepinto and node.type is not None and node.type.is_cfunction and getattr(node, 'is_called', False) and (node.entry.func_cname is not None):
            attrs = dict(name=node.entry.func_cname)
            self.tb.start('StepIntoFunction', attrs=attrs)
            self.tb.end('StepIntoFunction')
        self.visitchildren(node)
        return node

    def serialize_modulenode_as_function(self, node):
        """
        Serialize the module-level code as a function so the debugger will know
        it's a "relevant frame" and it will know where to set the breakpoint
        for 'break modulename'.
        """
        self._serialize_modulenode_as_function(node, dict(name=node.full_module_name.rpartition('.')[-1], cname=node.module_init_func_cname(), pf_cname='', qualified_name='', lineno='1', is_initmodule_function='True'))

    def _serialize_modulenode_as_function(self, node, attrs):
        self.tb.start('Function', attrs=attrs)
        self.tb.start('Locals')
        self.serialize_local_variables(node.scope.entries)
        self.tb.end('Locals')
        self.tb.start('Arguments')
        self.tb.end('Arguments')
        self.tb.start('StepIntoFunctions')
        self.register_stepinto = True
        self.visitchildren(node)
        self.register_stepinto = False
        self.tb.end('StepIntoFunctions')
        self.tb.end('Function')

    def serialize_local_variables(self, entries):
        for entry in entries.values():
            if not entry.cname:
                continue
            if entry.type.is_pyobject:
                vartype = 'PythonObject'
            else:
                vartype = 'CObject'
            if entry.from_closure:
                cname = '%s->%s' % (Naming.cur_scope_cname, entry.outer_entry.cname)
                qname = '%s.%s.%s' % (entry.scope.outer_scope.qualified_name, entry.scope.name, entry.name)
            elif entry.in_closure:
                cname = '%s->%s' % (Naming.cur_scope_cname, entry.cname)
                qname = entry.qualified_name
            else:
                cname = entry.cname
                qname = entry.qualified_name
            if not entry.pos:
                lineno = '0'
            else:
                lineno = str(entry.pos[1])
            attrs = dict(name=entry.name, cname=cname, qualified_name=qname, type=vartype, lineno=lineno)
            self.tb.start('LocalVar', attrs)
            self.tb.end('LocalVar')