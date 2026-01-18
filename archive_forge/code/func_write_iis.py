import abc
import logging
from pyomo.environ import SolverFactory
def write_iis(pyomo_model, iis_file_name, solver=None, logger=logger):
    """
    Write an irreducible infeasible set for a Pyomo MILP or LP
    using the specified commercial solver.

    Arguments
    ---------
        pyomo_model:
            A Pyomo Block or ConcreteModel
        iis_file_name:str
            A file name to write the IIS to, e.g., infeasible_model.ilp
        solver:str
            Specify the solver to use, one of "cplex", "gurobi", or "xpress".
            If None, the tool will use the first solver available.
        logger:logging.Logger
            A logger for messages. Uses pyomo.contrib.iis logger by default.

    Returns
    -------
        iis_file_name:str
            The file containing the IIS.
    """
    available_solvers = [s for s, sp in zip(_supported_solvers, _supported_solvers_persistent) if SolverFactory(sp).available(exception_flag=False)]
    if solver is None:
        if len(available_solvers) == 0:
            raise RuntimeError(f'Could not find a solver to use, supported solvers are {_supported_solvers}')
        solver = available_solvers[0]
        logger.info(f'Using solver {solver}')
    else:
        solver = solver.lower()
        solver = _remove_suffix(solver, '_persistent')
        if solver not in available_solvers:
            raise RuntimeError(f'The Pyomo persistent interface to {solver} could not be found.')
    solver_name = solver
    solver = SolverFactory(solver + '_persistent')
    solver.set_instance(pyomo_model, symbolic_solver_labels=True)
    iis = IISFactory(solver)
    iis.compute()
    iis_file_name = iis.write(iis_file_name)
    logger.info(f'IIS written to {iis_file_name}')
    return iis_file_name