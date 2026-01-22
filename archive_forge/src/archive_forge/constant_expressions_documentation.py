from pythran.analyses.aliases import Aliases
from pythran.analyses.globals_analysis import Globals
from pythran.analyses.locals_analysis import Locals
from pythran.analyses.pure_functions import PureFunctions
from pythran.intrinsic import FunctionIntr
from pythran.passmanager import NodeAnalysis
import gast as ast
Identify constant expressions.