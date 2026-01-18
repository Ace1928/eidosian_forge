from pythran.analyses.aliases import Aliases
from pythran.analyses.intrinsics import Intrinsics
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
from pythran.graph import DiGraph
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce

        Initialise globals effects as this analyse is inter-procedural.

        Initialisation done for Pythonic functions and default value set for
        user defined functions.
        