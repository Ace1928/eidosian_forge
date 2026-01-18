from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
def visit_Interactive(self, node):
    raise PythranSyntaxError('Interactive session not supported', node)