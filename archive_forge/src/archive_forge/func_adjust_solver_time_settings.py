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
def adjust_solver_time_settings(timing_data_obj, solver, config):
    """
    Adjust solver max time setting based on current PyROS elapsed
    time.

    Parameters
    ----------
    timing_data_obj : Bunch
        PyROS timekeeper.
    solver : solver type
        Solver for which to adjust the max time setting.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    original_max_time_setting : float or None
        If IPOPT or BARON is used, a float is returned.
        If GAMS is used, the ``options.add_options`` attribute
        of ``solver`` is returned.
        Otherwise, None is returned.
    custom_setting_present : bool or None
        If IPOPT or BARON is used, True if the max time is
        specified, False otherwise.
        If GAMS is used, True if the attribute ``options.add_options``
        is not None, False otherwise.
        If ``config.time_limit`` is None, then None is returned.

    Note
    ----
    (1) Adjustment only supported for GAMS, BARON, and IPOPT
        interfaces. This routine can be generalized to other solvers
        after a generic interface to the time limit setting
        is introduced.
    (2) For IPOPT, and probably also BARON, the CPU time limit
        rather than the wallclock time limit, is adjusted, as
        no interface to wallclock limit available.
        For this reason, extra 30s is added to time remaining
        for subsolver time limit.
        (The extra 30s is large enough to ensure solver
        elapsed time is not beneath elapsed time - user time limit,
        but not so large as to overshoot the user-specified time limit
        by an inordinate margin.)
    """
    if config.time_limit is not None:
        time_remaining = config.time_limit - get_main_elapsed_time(timing_data_obj)
        if isinstance(solver, type(SolverFactory('gams', solver_io='shell'))):
            original_max_time_setting = solver.options['add_options']
            custom_setting_present = 'add_options' in solver.options
            reslim_str = f'option reslim={max(30, 30 + time_remaining)};'
            if isinstance(solver.options['add_options'], list):
                solver.options['add_options'].append(reslim_str)
            else:
                solver.options['add_options'] = [reslim_str]
        else:
            if isinstance(solver, SolverFactory.get_class('baron')):
                options_key = 'MaxTime'
            elif isinstance(solver, SolverFactory.get_class('ipopt')):
                options_key = 'max_cpu_time'
            else:
                options_key = None
            if options_key is not None:
                custom_setting_present = options_key in solver.options
                original_max_time_setting = solver.options[options_key]
                solver.options[options_key] = max(30, 30 + time_remaining)
            else:
                custom_setting_present = False
                original_max_time_setting = None
                config.progress_logger.warning(f'Subproblem time limit setting not adjusted for subsolver of type:\n    {type(solver)}.\n    PyROS time limit may not be honored ')
        return (original_max_time_setting, custom_setting_present)
    else:
        return (None, None)