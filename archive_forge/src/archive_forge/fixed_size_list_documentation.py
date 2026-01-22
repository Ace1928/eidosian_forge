from pythran.passmanager import FunctionAnalysis
from pythran.tables import MODULES
import gast as ast

Whether a list usage makes it a candidate for fixed-size-list

This could be a type information, but it seems easier to implement it that way
