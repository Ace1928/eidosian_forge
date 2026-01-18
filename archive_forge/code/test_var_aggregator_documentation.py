import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
Test for transitivity in a variable equality set.