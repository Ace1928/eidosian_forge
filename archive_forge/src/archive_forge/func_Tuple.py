import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def Tuple(of_types):
    return Collection(Traits([TupleTrait(of_types), LenTrait, SliceTrait]), Integer(), TypeVariable(), TypeVariable())