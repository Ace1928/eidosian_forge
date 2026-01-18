import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest

        This matrix decomposes into 2x2 blocks
        |x x      |
        |x x      |
        |  x x x  |
        |    x x  |
        |      x x|
        