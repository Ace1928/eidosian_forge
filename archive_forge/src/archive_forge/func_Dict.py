import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def Dict(key_type, value_type):
    return Collection(Traits([DictTrait, LenTrait, NoSliceTrait]), key_type, value_type, key_type)