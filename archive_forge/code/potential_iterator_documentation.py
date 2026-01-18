from pythran.analyses.aliases import Aliases
from pythran.analyses.argument_read_once import ArgumentReadOnce
from pythran.passmanager import NodeAnalysis
import gast as ast
Find whether an expression can be replaced with an iterator.