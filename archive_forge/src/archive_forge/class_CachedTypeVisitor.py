from pythran.analyses import LocalNodeDeclarations, GlobalDeclarations, Scope
from pythran.analyses import YieldPoints, IsAssigned, ASTMatcher, AST_any
from pythran.analyses import RangeValues, PureExpressions, Dependencies
from pythran.analyses import Immediates, Ancestors, StrictAliases
from pythran.config import cfg
from pythran.cxxgen import Template, Include, Namespace, CompilationUnit
from pythran.cxxgen import Statement, Block, AnnotatedStatement, Typedef, Label
from pythran.cxxgen import Value, FunctionDeclaration, EmptyStatement, Nop
from pythran.cxxgen import FunctionBody, Line, ReturnStatement, Struct, Assign
from pythran.cxxgen import For, While, TryExcept, ExceptHandler, If, AutoFor
from pythran.cxxgen import StatementWithComments
from pythran.openmp import OMPDirective
from pythran.passmanager import Backend
from pythran.syntax import PythranSyntaxError
from pythran.tables import operator_to_lambda, update_operator_to_lambda
from pythran.tables import pythran_ward, attributes as attributes_table
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE, TYPE_TO_SUFFIX
from pythran.types.types import Types
from pythran.utils import attr_to_path, pushpop, cxxid, isstr, isnum
from pythran.utils import isextslice, ispowi, quote_cxxstring
from pythran import metadata, unparse
from math import isnan, isinf
import gast as ast
from functools import reduce
import io
class CachedTypeVisitor:

    def __init__(self, other=None):
        if other is None:
            self.rcache = dict()
            self.mapping = dict()
        else:
            self.rcache = other.rcache.copy()
            self.mapping = other.mapping.copy()

    def __call__(self, node):
        if node not in self.mapping:
            t = node.generate(self)
            if node not in self.mapping:
                if t in self.rcache:
                    self.mapping[node] = self.rcache[t]
                else:
                    self.rcache[t] = len(self.mapping)
                    self.mapping[node] = len(self.mapping)
        return '__type{0}'.format(self.mapping[node])

    def typedefs(self):
        kv = sorted(self.rcache.items(), key=lambda x: x[1])
        L = list()
        for k, v in kv:
            typename = '__type' + str(v)
            L.append(Typedef(Value(k, typename)))
        return L