import os
from filecmp import cmp
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.collections import OrderedSet
from pyomo.common.fileutils import this_file_dir
import pyomo.core.expr as EXPR
from pyomo.core.base import SymbolMap
from pyomo.environ import (
from pyomo.repn.plugins.baron_writer import expression_to_string
These tests verified that the BARON writer complained loudly for
    variables that were not on the model, not on an active block, or not
    on a Block ctype.  As we are relaxing that requirement throughout
    Pyomo, these tests have been disabled.