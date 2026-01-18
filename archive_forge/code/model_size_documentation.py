import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
Retrieve generator of activated disjuncts on disjunction.