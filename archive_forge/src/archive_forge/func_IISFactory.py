import abc
import logging
from pyomo.environ import SolverFactory
def IISFactory(solver):
    if solver.name not in _solver_map:
        raise RuntimeError(f'Unrecognized solver {solver.name}')
    return _solver_map[solver.name](solver)