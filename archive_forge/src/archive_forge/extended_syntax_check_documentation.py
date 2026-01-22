from pythran.passmanager import ModuleAnalysis
from pythran.analyses import StrictAliases, ArgumentEffects
from pythran.syntax import PythranSyntaxError
from pythran.intrinsic import ConstantIntr, FunctionIntr
from pythran import metadata
import gast as ast

    Perform advanced syntax checking, based on strict aliases analysis:
    - is there a function redefinition?
    - is there a function call that does not match the called expression arity?
    - is there an operation that updates a global variable?
    