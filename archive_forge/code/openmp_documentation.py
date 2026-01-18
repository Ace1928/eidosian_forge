from pythran.passmanager import Transformation
import pythran.metadata as metadata
from pythran.spec import parse_pytypes
from pythran.types.conversion import pytype_to_ctype
from pythran.utils import isstr
from gast import AST
import gast as ast
import re
A simple contextual "parser" for an OpenMP string