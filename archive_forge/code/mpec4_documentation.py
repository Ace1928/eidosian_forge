import logging
from pyomo.core.base import (
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct

        Convert a common form that can processed by AMPL
        