import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def _c_rule(m, a):
    return m.B[a].PORT == m.B[(a + 1) % 2].PORT