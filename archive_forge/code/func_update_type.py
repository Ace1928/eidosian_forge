from pythran.analyses import LazynessAnalysis, StrictAliases, YieldPoints
from pythran.analyses import LocalNodeDeclarations, Immediates, RangeValues
from pythran.analyses import Ancestors
from pythran.config import cfg
from pythran.cxxtypes import TypeBuilder, ordered_set
from pythran.intrinsic import UserFunction, Class
from pythran.passmanager import ModuleAnalysis
from pythran.tables import operator_to_lambda, MODULES
from pythran.types.conversion import pytype_to_ctype
from pythran.types.reorder import Reorder
from pythran.utils import attr_to_path, cxxid, isnum, isextslice
from collections import defaultdict
from functools import reduce
import gast as ast
from itertools import islice
from copy import deepcopy
def update_type(self, node, ty_builder, *args):
    if ty_builder is None:
        ty, = args
    elif any((arg is self.builder.UnknownType for arg in args)):
        ty = self.builder.UnknownType
    else:
        ty = ty_builder(*args)
    curr_ty = self.result.get(node, self.builder.UnknownType)
    if isinstance(curr_ty, tuple):
        return
    self.result[node] = self.combined(curr_ty, ty)