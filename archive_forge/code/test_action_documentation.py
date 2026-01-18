from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
Check the use of the __getattr__ method