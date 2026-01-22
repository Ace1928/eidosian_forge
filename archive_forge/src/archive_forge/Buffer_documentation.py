from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab

    Output the run-time type information (__Pyx_TypeInfo) for given dtype,
    and return the name of the type info struct.

    Structs with two floats of the same size are encoded as complex numbers.
    One can separate between complex numbers declared as struct or with native
    encoding by inspecting to see if the fields field of the type is
    filled in.
    