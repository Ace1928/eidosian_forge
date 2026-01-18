import logging
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct, GDP_Error, Disjunction
from pyomo.core import TraversalStrategy, TransformationFactory
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.common.deprecation import deprecated
Reclassify Disjuncts to Blocks.

    HACK: this will reclassify all Disjuncts to Blocks so the current writers
    can find the variables

    