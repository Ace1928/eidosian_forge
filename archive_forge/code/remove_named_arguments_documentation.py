from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import MODULES
import gast as ast
from copy import deepcopy

        Gather keywords to positional argument information

        Assumes the named parameter exist, raises a KeyError otherwise
        