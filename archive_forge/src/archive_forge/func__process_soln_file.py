import logging
import os
import subprocess
import re
import tempfile
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
def _process_soln_file(self, results, TimFile, INPUT):
    extract_marginals = False
    extract_price = False
    for suffix in self._suffixes:
        flag = False
        if re.match(suffix, 'rc'):
            extract_marginals = True
            flag = True
        if re.match(suffix, 'dual'):
            extract_price = True
            flag = True
        if not flag:
            raise RuntimeError('***The BARON solver plugin cannotextract solution suffix=' + suffix)
    soln = Solution()
    line = TimFile.readline().split()
    try:
        results.problem.name = line[0]
        results.problem.number_of_constraints = int(line[1])
        results.problem.number_of_variables = int(line[2])
        try:
            results.problem.lower_bound = float(line[5])
        except ValueError:
            results.problem.lower_bound = float('-inf')
        try:
            results.problem.upper_bound = float(line[6])
        except ValueError:
            results.problem.upper_bound = float('inf')
        results.problem.missing_bounds = line[9]
        results.problem.iterations = line[10]
        results.problem.node_opt = line[11]
        results.problem.node_memmax = line[12]
        results.problem.cpu_time = float(line[13])
        results.problem.wall_time = float(line[14])
    except IndexError:
        pass
    soln.gap = results.problem.upper_bound - results.problem.lower_bound
    solver_status = line[7]
    model_status = line[8]
    objective = None
    objective_label = '__default_objective__'
    soln.objective[objective_label] = {'Value': None}
    results.problem.number_of_objectives = 1
    if objective is not None:
        results.problem.sense = 'minimizing' if objective.is_minimizing() else 'maximizing'
    if solver_status == '1':
        results.solver.status = SolverStatus.ok
    elif solver_status == '2':
        results.solver.status = SolverStatus.error
        results.solver.termination_condition = TerminationCondition.error
        results.solver.termination_message = 'Insufficient memory to store the number of nodes required for this search tree. Increase physical memory or change algorithmic options'
    elif solver_status == '3':
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.maxIterations
    elif solver_status == '4':
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.maxTimeLimit
    elif solver_status == '5':
        results.solver.status = SolverStatus.warning
        results.solver.termination_condition = TerminationCondition.other
    elif solver_status == '6':
        results.solver.status = SolverStatus.aborted
        results.solver.termination_condition = TerminationCondition.userInterrupt
    elif solver_status == '7':
        results.solver.status = SolverStatus.error
        results.solver.termination_condition = TerminationCondition.error
    elif solver_status == '8':
        results.solver.status = SolverStatus.unknown
        results.solver.termination_condition = TerminationCondition.unknown
    elif solver_status == '9':
        results.solver.status = SolverStatus.error
        results.solver.termination_condition = TerminationCondition.solverFailure
    elif solver_status == '10':
        results.solver.status = SolverStatus.error
        results.solver.termination_condition = TerminationCondition.error
    elif solver_status == '11':
        results.solver.status = SolverStatus.aborted
        results.solver.termination_condition = TerminationCondition.licensingProblems
        results.solver.termination_message = 'Run terminated because of a licensing error.'
    if model_status == '1':
        soln.status = SolutionStatus.optimal
        results.solver.termination_condition = TerminationCondition.optimal
    elif model_status == '2':
        soln.status = SolutionStatus.infeasible
        results.solver.termination_condition = TerminationCondition.infeasible
    elif model_status == '3':
        soln.status = SolutionStatus.unbounded
        results.solver.termination_condition = TerminationCondition.unbounded
    elif model_status == '4':
        soln.status = SolutionStatus.feasible
    elif model_status == '5':
        soln.status = SolutionStatus.unknown
    if results.solver.status not in [SolverStatus.error, SolverStatus.aborted]:
        var_value = []
        var_name = []
        var_marginal = []
        con_price = []
        SolvedDuringPreprocessing = False
        line = '\n'
        while line and '***' not in line:
            line = INPUT.readline()
            if 'Problem solved during preprocessing' in line:
                SolvedDuringPreprocessing = True
        INPUT.readline()
        INPUT.readline()
        try:
            objective_value = float(INPUT.readline().split()[4])
        except IndexError:
            if solver_status == '1' and model_status in ('1', '4'):
                logger.error("Failed to process BARON solution file: could not extract the final\nobjective value, but BARON completed normally.  This is indicative of a\nbug in Pyomo's BARON solution parser.  Please report this (along with\nthe Pyomo model and BARON version) to the Pyomo Developers.")
            return
        INPUT.readline()
        INPUT.readline()
        line = INPUT.readline()
        while line.strip() != '':
            var_value.append(float(line.split()[2]))
            line = INPUT.readline()
        has_dual_info = False
        if 'Corresponding dual solution vector is' in INPUT.readline():
            has_dual_info = True
            INPUT.readline()
            line = INPUT.readline()
            while 'Price' not in line and line.strip() != '':
                var_marginal.append(float(line.split()[2]))
                line = INPUT.readline()
            if 'Price' in line:
                line = INPUT.readline()
                line = INPUT.readline()
                while line.strip() != '':
                    con_price.append(float(line.split()[2]))
                    line = INPUT.readline()
        while 'The best solution found is' not in INPUT.readline():
            pass
        INPUT.readline()
        INPUT.readline()
        line = INPUT.readline()
        while line.strip() != '':
            var_name.append(line.split()[0])
            line = INPUT.readline()
        assert len(var_name) == len(var_value)
        soln_variable = soln.variable
        for i, (label, val) in enumerate(zip(var_name, var_value)):
            soln_variable[label] = {'Value': val}
            if extract_marginals and has_dual_info:
                soln_variable[label]['rc'] = var_marginal[i]
        if extract_price and has_dual_info:
            soln_constraint = soln.constraint
            for i, price_val in enumerate(con_price, 1):
                con_label = '.c' + str(i)
                soln_constraint[con_label] = {'dual': price_val}
        if not (SolvedDuringPreprocessing and soln.status == SolutionStatus.infeasible):
            soln.objective[objective_label] = {'Value': objective_value}
        results.solution.insert(soln)