from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
class ContainerOf(object):
    """
    Represents a container of something

    We just know that if indexed by the integer value `index',
    we get `containees'
    """
    UnknownIndex = float('nan')
    __slots__ = ('index', 'containees')

    def __init__(self, containees, index=UnknownIndex):
        self.index = index
        self.containees = containees