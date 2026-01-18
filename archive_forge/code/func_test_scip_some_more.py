import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
import random
@unittest.skipIf(not scip_available, 'SCIP solver is not available.')
def test_scip_some_more():
    list_concrete_models = [problem_lp_unbounded(), problem_lp_infeasible(), problem_lp_optimal(), problem_milp_unbounded(), problem_milp_infeasible(), problem_milp_optimal(), problem_milp_feasible()]
    list_extra_data_expected = [(), (), ('Time', 'Gap', 'Primal bound', 'Dual bound'), (), (), ('Time', 'Gap', 'Primal bound', 'Dual bound'), ('Time', 'Gap', 'Primal bound', 'Dual bound')]
    solver_timelimit = 1
    solver_abs_mip_gap = 0
    solver_rel_mip_gap = 1e-06
    for problem_index, problem in enumerate(list_concrete_models):
        print('******************************')
        print('******************************')
        print(problem.name)
        print('******************************')
        print('******************************')
        results, opt = optimise(problem, solver_timelimit, solver_rel_mip_gap, solver_abs_mip_gap, print_solver_output=True)
        print(results)
        executable = opt._command.cmd[0]
        version = opt._known_versions[executable]
        if version < (8, 0, 0, 0):
            continue
        for log_file_attr in list_extra_data_expected[problem_index]:
            assert log_file_attr in results['Solver'][0]