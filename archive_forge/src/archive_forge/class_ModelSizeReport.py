import logging
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.core import Block, Constraint, Var
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction
class ModelSizeReport(Bunch):
    """Stores model size information.

    Activated blocks are those who have an active flag of True and whose
    parent, if exists, is an activated block or an activated Disjunct.

    Activated constraints are those with an active flag of True and: are
    reachable via an activated Block, are on an activated Disjunct, or are on a
    disjunct with indicator_var fixed to 1 with active flag True.

    Activated variables refer to the presence of the variable on an activated
    constraint, or that the variable is an indicator_var for an activated
    Disjunct.

    Activated disjuncts refer to disjuncts with an active flag of True, have an
    unfixed indicator_var, and who participate in an activated Disjunction.

    Activated disjunctions follow the same rules as activated constraints.

    """
    pass