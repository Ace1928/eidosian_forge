import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def _format_record_attr(self, attr_name):
    """Format attribute record for logging."""
    attr_val = getattr(self, attr_name)
    if attr_val is None:
        fmt_str = f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}s'
        return f'{'-':{fmt_str}}'
    else:
        attr_val_fstrs = {'iteration': "f'{attr_val:d}'", 'objective': "f'{attr_val: .4e}'", 'first_stage_var_shift': "f'{attr_val:.4e}'", 'second_stage_var_shift': "f'{attr_val:.4e}'", 'dr_var_shift': "f'{attr_val:.4e}'", 'num_violated_cons': "f'{attr_val:d}'", 'max_violation': "f'{attr_val:.4e}'", 'elapsed_time': "f'{attr_val:.3f}'"}
        if attr_name in ['second_stage_var_shift', 'dr_var_shift']:
            qual = '*' if not self.dr_polishing_success else ''
        elif attr_name == 'num_violated_cons':
            qual = '+' if not self.all_sep_problems_solved else ''
        elif attr_name == 'max_violation':
            qual = 'g' if self.global_separation else ''
        else:
            qual = ''
        attr_val_str = f'{eval(attr_val_fstrs[attr_name])}{qual}'
        return f'{attr_val_str:{f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}'}}'