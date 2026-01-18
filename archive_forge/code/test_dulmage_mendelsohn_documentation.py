import pyomo.common.unittest as unittest
from pyomo.common.dependencies import networkx as nx, networkx_available
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (

        Graph with the following incidence matrix:
        |x x         x|
        |x   x        |
        |      x   x  |
        |        x x  |
        |      x x    |
        |            x|
        |            x|
        