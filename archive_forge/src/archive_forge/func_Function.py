import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def Function(from_types, to_type):
    """A binary type constructor which builds function types"""
    return TypeOperator('fun', list(from_types) + [to_type])