import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentMap, Bunch
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver, BranchDirection
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.solvers.mockmip import MockMIP
from pyomo.core.base import Var, Suffix, active_export_suffix_generator
from pyomo.core.kernel.suffix import export_suffix_generator
from pyomo.core.kernel.block import IBlock
from pyomo.util.components import iter_component
@SolverFactory.register('_cplex_shell', doc='Shell interface to the CPLEX LP/MIP solver')
class CPLEXSHELL(ILMLicensedSystemCallSolver):
    """Shell interface to the CPLEX LP/MIP solver"""

    def __init__(self, **kwds):
        kwds['type'] = 'cplex'
        ILMLicensedSystemCallSolver.__init__(self, **kwds)
        self._warm_start_solve = False
        self._warm_start_file_name = None
        self._valid_problem_formats = [ProblemFormat.cpxlp, ProblemFormat.mps]
        self._valid_result_formats = {}
        self._valid_result_formats[ProblemFormat.cpxlp] = [ResultsFormat.soln]
        self._valid_result_formats[ProblemFormat.mps] = [ResultsFormat.soln]
        self.set_problem_format(ProblemFormat.cpxlp)
        self._capabilities = Bunch()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def warm_start_capable(self):
        return True

    def _warm_start(self, instance):
        output_index = 0
        if isinstance(instance, IBlock):
            smap = getattr(instance, '._symbol_maps')[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        with open(self._warm_start_file_name, 'w') as mst_file:
            mst_file.write('<?xml version="1.0" ?>\n')
            mst_file.write('<CPLEXSolution version="1.0">\n')
            mst_file.write('<header/>\n')
            mst_file.write('<quality/>\n')
            mst_file.write('<variables>\n')
            for var in instance.component_data_objects(Var):
                if var.value is not None and id(var) in byObject:
                    name = byObject[id(var)]
                    mst_file.write('<variable index="%d" name="%s" value="%f" />\n' % (output_index, name, var.value))
                    output_index = output_index + 1
            mst_file.write('</variables>\n')
            mst_file.write('</CPLEXSolution>\n')
    SUFFIX_PRIORITY_NAME = 'priority'
    SUFFIX_DIRECTION_NAME = 'direction'

    def _write_priorities_file(self, instance):
        """Write a variable priorities file in the CPLEX ORD format."""
        priorities, directions = self._get_suffixes(instance)
        rows = self._convert_priorities_to_rows(instance, priorities, directions)
        self._write_priority_rows(rows)

    def _get_suffixes(self, instance):
        if isinstance(instance, IBlock):
            suffixes = ((suf.name, suf) for suf in export_suffix_generator(instance, datatype=Suffix.INT, active=True, descend_into=False))
        else:
            suffixes = active_export_suffix_generator(instance, datatype=Suffix.INT)
        suffixes = dict(suffixes)
        if self.SUFFIX_PRIORITY_NAME not in suffixes:
            raise ValueError('Cannot write branching priorities file as `model.%s` Suffix has not been declared.' % (self.SUFFIX_PRIORITY_NAME,))
        return (suffixes[self.SUFFIX_PRIORITY_NAME], suffixes.get(self.SUFFIX_DIRECTION_NAME, ComponentMap()))

    def _convert_priorities_to_rows(self, instance, priorities, directions):
        if isinstance(instance, IBlock):
            smap = getattr(instance, '._symbol_maps')[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        rows = []
        for var, priority in priorities.items():
            if priority is None or not var.active:
                continue
            if not 0 <= priority == int(priority):
                raise ValueError('`priority` must be a non-negative integer')
            var_direction = directions.get(var, BranchDirection.default)
            for child_var in iter_component(var):
                if id(child_var) not in byObject:
                    continue
                child_var_direction = directions.get(child_var, var_direction)
                rows.append((byObject[id(child_var)], priority, child_var_direction))
        return rows

    def _write_priority_rows(self, rows):
        with open(self._priorities_file_name, 'w') as ord_file:
            ord_file.write(ORDFileSchema.HEADER)
            for var_name, priority, direction in rows:
                ord_file.write(ORDFileSchema.ROW(var_name, priority, direction))
            ord_file.write(ORDFileSchema.FOOTER)

    def _presolve(self, *args, **kwds):
        TempfileManager.push()
        self._warm_start_solve = kwds.pop('warmstart', False)
        self._warm_start_file_name = _validate_file_name(self, kwds.pop('warmstart_file', None), 'warm start')
        user_warmstart = self._warm_start_file_name is not None
        if self._warm_start_solve and isinstance(args[0], str):
            pass
        elif self._warm_start_solve and (not isinstance(args[0], str)):
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = TempfileManager.create_tempfile(suffix='.cplex.mst')
        self._priorities_solve = kwds.pop('priorities', False)
        self._priorities_file_name = _validate_file_name(self, kwds.pop('priorities_file', None), 'branching priorities')
        user_priorities = self._priorities_file_name is not None
        if self._priorities_solve and (not isinstance(args[0], str)) and (not user_priorities):
            self._priorities_file_name = TempfileManager.create_tempfile(suffix='.cplex.ord')
        ILMLicensedSystemCallSolver._presolve(self, *args, **kwds)
        if len(args) > 0 and (not isinstance(args[0], str)):
            if len(args) != 1:
                raise ValueError('CPLEX _presolve method can only handle a single problem instance - %s were supplied' % (len(args),))
            if self._warm_start_solve and (not user_warmstart):
                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing:
                    print('Warm start write time= %.2f seconds' % (end_time - start_time))
            if self._priorities_solve and (not user_priorities):
                start_time = time.time()
                self._write_priorities_file(args[0])
                end_time = time.time()
                if self._report_timing:
                    print('Branching priorities write time= %.2f seconds' % (end_time - start_time))

    def _default_executable(self):
        executable = Executable('cplex')
        if not executable:
            logger.warning("Could not locate the 'cplex' executable, which is required for solver %s" % self.name)
            self.enable = False
            return None
        return executable.path()

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if solver_exec is None:
            return _extract_version('')
        results = subprocess.run([solver_exec, '-c', 'quit'], timeout=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return _extract_version(results.stdout)

    def create_command_line(self, executable, problem_files):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.cplex.log')
        self._log_file = _validate_file_name(self, self._log_file, 'log')
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix='.cplex.sol')
        self._soln_file = _validate_file_name(self, self._soln_file, 'solution')
        script = 'set logfile %s\n' % (self._log_file,)
        if self._timelimit is not None and self._timelimit > 0.0:
            script += 'set timelimit %s\n' % (self._timelimit,)
        if self.options.mipgap is not None and float(self.options.mipgap) > 0.0:
            script += 'set mip tolerances mipgap %s\n' % (self.options.mipgap,)
        for key in self.options:
            if key == 'relax_integrality' or key == 'mipgap':
                continue
            elif isinstance(self.options[key], str) and ' ' in self.options[key]:
                opt = ' '.join(key.split('_')) + ' ' + str(self.options[key])
            else:
                opt = ' '.join(key.split('_')) + ' ' + str(self.options[key])
            script += 'set %s\n' % (opt,)
        _lp_file = _validate_file_name(self, problem_files[0], 'LP')
        script += 'read %s\n' % (_lp_file,)
        if self._warm_start_solve and self._warm_start_file_name is not None:
            script += 'read %s\n' % (self._warm_start_file_name,)
        if self._priorities_solve and self._priorities_file_name is not None:
            script += 'read %s\n' % (self._priorities_file_name,)
        if 'relax_integrality' in self.options:
            script += 'change problem lp\n'
        script += 'display problem stats\n'
        script += 'optimize\n'
        script += 'write %s\n' % (self._soln_file,)
        script += 'quit\n'
        if self._keepfiles:
            script_fname = TempfileManager.create_tempfile(suffix='.cplex.script')
            tmp = open(script_fname, 'w')
            tmp.write(script)
            tmp.close()
            print('Solver script file=' + script_fname)
            if self._warm_start_solve and self._warm_start_file_name is not None:
                print('Solver warm-start file=' + self._warm_start_file_name)
            if self._priorities_solve and self._priorities_file_name is not None:
                print('Solver priorities file=' + self._priorities_file_name)
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, script=script, log_file=self._log_file, env=None)

    def process_logfile(self):
        """
        Process logfile
        """
        results = SolverResults()
        results.problem.number_of_variables = None
        results.problem.number_of_nonzeros = None
        OUTPUT = open(self._log_file)
        output = ''.join(OUTPUT.readlines())
        OUTPUT.close()
        cplex_version = None
        self._best_bound = None
        self._gap = None
        for line in output.split('\n'):
            tokens = re.split('[ \t]+', line.strip())
            if len(tokens) > 3 and tokens[0] == 'CPLEX' and (tokens[1] == 'Error'):
                results.solver.status = SolverStatus.error
                results.solver.error = ' '.join(tokens)
            elif len(tokens) >= 3 and tokens[0] == 'ILOG' and (tokens[1] == 'CPLEX'):
                cplex_version = tokens[2].rstrip(',')
            elif len(tokens) >= 3 and tokens[0] == 'Variables':
                if results.problem.number_of_variables is None:
                    results.problem.number_of_variables = int(tokens[2])
            elif len(tokens) >= 4 and tokens[0] == 'Linear' and (tokens[1] == 'constraints'):
                results.problem.number_of_constraints = int(tokens[3])
            elif len(tokens) >= 3 and tokens[0] == 'Nonzeros':
                if results.problem.number_of_nonzeros is None:
                    results.problem.number_of_nonzeros = int(tokens[2])
            elif len(tokens) >= 5 and tokens[4] == 'MINIMIZE':
                results.problem.sense = ProblemSense.minimize
            elif len(tokens) >= 5 and tokens[4] == 'MAXIMIZE':
                results.problem.sense = ProblemSense.maximize
            elif len(tokens) >= 4 and tokens[0] == 'Solution' and (tokens[1] == 'time') and (tokens[2] == '='):
                results.solver.user_time = float(tokens[3])
            elif len(tokens) >= 4 and tokens[0] == 'Primal' and (tokens[1] == 'simplex') and (tokens[3] == 'Optimal:'):
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == 'Dual' and (tokens[1] == 'simplex') and (tokens[3] == 'Optimal:'):
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == 'Barrier' and (tokens[2] == 'Optimal:'):
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == 'Dual' and (tokens[3] == 'Infeasible:'):
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 4 and tokens[0] == 'MIP' and (tokens[2] == 'Integer') and (tokens[3] == 'infeasible.'):
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 10 and tokens[0] == 'MIP' and (tokens[2] == 'Time') and (tokens[3] == 'limit') and (tokens[6] == 'feasible:'):
                results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.maxTimeLimit
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 10 and tokens[0] == 'Current' and (tokens[1] == 'MIP') and (tokens[2] == 'best') and (tokens[3] == 'bound'):
                self._best_bound = float(tokens[5])
                self._gap = float(tokens[8].rstrip(','))
            elif len(tokens) >= 4 and tokens[0] == 'MIP' and (tokens[2] == 'Integer') and (tokens[3] == 'optimal') or (len(tokens) >= 4 and tokens[0] == 'MIP' and (tokens[2] == 'Integer') and (tokens[3] == 'optimal,')):
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) >= 3 and tokens[0] == 'Presolve' and (tokens[2] == 'Infeasible.'):
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.termination_message = ' '.join(tokens)
            elif len(tokens) == 6 and tokens[2] == 'Integer' and (tokens[3] == 'infeasible') and (tokens[5] == 'unbounded.') or (len(tokens) >= 4 and tokens[0] == 'MIP' and (tokens[1] == '-') and (tokens[2] == 'Integer') and (tokens[3] == 'unbounded:')) or (len(tokens) >= 5 and tokens[0] == 'Presolve' and (tokens[2] == 'Unbounded') and (tokens[4] == 'infeasible.')):
                if results.solver.status == SolverStatus.error:
                    results.solver.status = SolverStatus.warning
                else:
                    results.solver.status = SolverStatus.ok
                results.solver.termination_condition = TerminationCondition.unbounded
                results.solver.termination_message = ' '.join(tokens)
        try:
            if isinstance(results.solver.termination_message, str):
                results.solver.termination_message = results.solver.termination_message.replace(':', '\\x3a')
        except:
            pass
        return results

    def process_soln_file(self, results):
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        extract_rc = False
        extract_lrc = False
        extract_urc = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, 'dual'):
                extract_duals = True
                flag = True
            if re.match(suffix, 'slack'):
                extract_slacks = True
                flag = True
            if re.match(suffix, 'rc'):
                extract_reduced_costs = True
                extract_rc = True
                flag = True
            if re.match(suffix, 'lrc'):
                extract_reduced_costs = True
                extract_lrc = True
                flag = True
            if re.match(suffix, 'urc'):
                extract_reduced_costs = True
                extract_urc = True
                flag = True
            if not flag:
                raise RuntimeError('***The CPLEX solver plugin cannot extract solution suffix=' + suffix)
        if not os.path.exists(self._soln_file):
            return
        range_duals = {}
        range_slacks = {}
        soln = Solution()
        soln.objective['__default_objective__'] = {'Value': None}
        soln_variables = soln.variable
        soln_constraints = soln.constraint
        INPUT = open(self._soln_file, 'r')
        results.problem.number_of_objectives = 1
        time_limit_exceeded = False
        mip_problem = False
        for line in INPUT:
            line = line.strip()
            line = line.lstrip('<?/')
            line = line.rstrip('/>?')
            tokens = line.split(' ')
            if tokens[0] == 'variable':
                variable_name = None
                variable_value = None
                variable_reduced_cost = None
                variable_status = None
                for i in range(1, len(tokens)):
                    field_name = tokens[i].split('=')[0]
                    field_value = tokens[i].split('=')[1].lstrip('"').rstrip('"')
                    if field_name == 'name':
                        variable_name = field_value
                    elif field_name == 'value':
                        variable_value = field_value
                    elif extract_reduced_costs is True and field_name == 'reducedCost':
                        variable_reduced_cost = field_value
                    elif extract_reduced_costs is True and field_name == 'status':
                        variable_status = field_value
                if variable_name != 'ONE_VAR_CONSTANT':
                    variable = soln_variables[variable_name] = {'Value': float(variable_value)}
                    if variable_reduced_cost is not None and extract_reduced_costs is True:
                        try:
                            if extract_rc is True:
                                variable['Rc'] = float(variable_reduced_cost)
                            if variable_status is not None:
                                if extract_lrc is True:
                                    if variable_status == 'LL':
                                        variable['Lrc'] = float(variable_reduced_cost)
                                    else:
                                        variable['Lrc'] = 0.0
                                if extract_urc is True:
                                    if variable_status == 'UL':
                                        variable['Urc'] = float(variable_reduced_cost)
                                    else:
                                        variable['Urc'] = 0.0
                        except:
                            raise ValueError('Unexpected reduced-cost value=' + str(variable_reduced_cost) + ' encountered for variable=' + variable_name)
            elif tokens[0] == 'constraint' and (extract_duals is True or extract_slacks is True):
                is_range = False
                rlabel = None
                rkey = None
                for i in range(1, len(tokens)):
                    field_name = tokens[i].split('=')[0]
                    field_value = tokens[i].split('=')[1].lstrip('"').rstrip('"')
                    if field_name == 'name':
                        if field_value.startswith('c_'):
                            constraint = soln_constraints[field_value] = {}
                        elif field_value.startswith('r_l_'):
                            is_range = True
                            rlabel = field_value[4:]
                            rkey = 0
                        elif field_value.startswith('r_u_'):
                            is_range = True
                            rlabel = field_value[4:]
                            rkey = 1
                    elif extract_duals is True and field_name == 'dual':
                        if is_range is False:
                            constraint['Dual'] = float(field_value)
                        else:
                            range_duals.setdefault(rlabel, [0, 0])[rkey] = float(field_value)
                    elif extract_slacks is True and field_name == 'slack':
                        if is_range is False:
                            constraint['Slack'] = float(field_value)
                        else:
                            range_slacks.setdefault(rlabel, [0, 0])[rkey] = float(field_value)
            elif tokens[0].startswith('problemName'):
                filename = tokens[0].split('=')[1].strip().lstrip('"').rstrip('"')
                results.problem.name = os.path.basename(filename)
                if '.' in results.problem.name:
                    results.problem.name = results.problem.name.split('.')[0]
                tINPUT = open(filename, 'r')
                for tline in tINPUT:
                    tline = tline.strip()
                    if tline == '':
                        continue
                    tokens = re.split('[\t ]+', tline)
                    if tokens[0][0] in ['\\', '*']:
                        continue
                    elif tokens[0] == 'NAME':
                        results.problem.name = tokens[1]
                    else:
                        sense = tokens[0].lower()
                        if sense in ['max', 'maximize']:
                            results.problem.sense = ProblemSense.maximize
                        if sense in ['min', 'minimize']:
                            results.problem.sense = ProblemSense.minimize
                    break
                tINPUT.close()
            elif tokens[0].startswith('objectiveValue') and tokens[0] != 'objectiveValues':
                objective_value = tokens[0].split('=')[1].strip().lstrip('"').rstrip('"')
                soln.objective['__default_objective__']['Value'] = float(objective_value)
            elif tokens[0] == 'objective':
                fields = {}
                for field in tokens[1:]:
                    k, v = field.split('=')
                    fields[k] = v.strip('"')
                soln.objective.setdefault(fields['name'], {})['Value'] = float(fields['value'])
            elif tokens[0].startswith('solutionStatusValue'):
                pieces = tokens[0].split('=')
                solution_status = eval(pieces[1])
                if soln.status == SolutionStatus.unknown:
                    if solution_status == 1:
                        soln.status = SolutionStatus.optimal
                    elif solution_status == 3:
                        soln.status = SolutionStatus.infeasible
                        soln.gap = None
                    else:
                        soln.status = SolutionStatus.error
                        soln.gap = None
            elif tokens[0].startswith('solutionStatusString'):
                solution_status = ' '.join(tokens).split('=')[1].strip().lstrip('"').rstrip('"')
                if solution_status in ['optimal', 'integer optimal solution', 'integer optimal, tolerance']:
                    soln.status = SolutionStatus.optimal
                    soln.gap = 0.0
                    results.problem.lower_bound = soln.objective['__default_objective__']['Value']
                    results.problem.upper_bound = soln.objective['__default_objective__']['Value']
                    if 'integer' in solution_status:
                        mip_problem = True
                elif solution_status in ['infeasible']:
                    soln.status = SolutionStatus.infeasible
                    soln.gap = None
                elif solution_status in ['time limit exceeded']:
                    time_limit_exceeded = True
            elif tokens[0].startswith('MIPNodes'):
                if mip_problem:
                    n = eval(eval(' '.join(tokens).split('=')[1].strip()).lstrip('"').rstrip('"'))
                    results.solver.statistics.branch_and_bound.number_of_created_subproblems = n
                    results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = n
            elif tokens[0].startswith('primalFeasible') and time_limit_exceeded is True:
                primal_feasible = int(' '.join(tokens).split('=')[1].strip().lstrip('"').rstrip('"'))
                if primal_feasible == 1:
                    soln.status = SolutionStatus.feasible
                    if results.problem.sense == ProblemSense.minimize:
                        results.problem.upper_bound = soln.objective['__default_objective__']['Value']
                    else:
                        results.problem.lower_bound = soln.objective['__default_objective__']['Value']
                else:
                    soln.status = SolutionStatus.infeasible
        if self._best_bound is not None:
            if results.problem.sense == ProblemSense.minimize:
                results.problem.lower_bound = self._best_bound
            else:
                results.problem.upper_bound = self._best_bound
        if self._gap is not None:
            soln.gap = self._gap
        for key, (ld, ud) in range_duals.items():
            if abs(ld) > abs(ud):
                soln_constraints['r_l_' + key] = {'Dual': ld}
            else:
                soln_constraints['r_l_' + key] = {'Dual': ud}
        for key, (ls, us) in range_slacks.items():
            if abs(ls) > abs(us):
                soln_constraints.setdefault('r_l_' + key, {})['Slack'] = ls
            else:
                soln_constraints.setdefault('r_l_' + key, {})['Slack'] = us
        if not results.solver.status is SolverStatus.error:
            if results.solver.termination_condition in [TerminationCondition.unknown, TerminationCondition.globallyOptimal, TerminationCondition.locallyOptimal, TerminationCondition.optimal, TerminationCondition.other]:
                results.solution.insert(soln)
            elif results.solver.termination_condition is TerminationCondition.maxTimeLimit and soln.status is not SolutionStatus.infeasible:
                results.solution.insert(soln)
        INPUT.close()

    def _postsolve(self):
        filename_list = os.listdir('.')
        for filename in filename_list:
            try:
                if re.match('cplex\\.log', filename) != None:
                    os.remove(filename)
                elif re.match('clone\\d+\\.log', filename) != None:
                    os.remove(filename)
            except OSError:
                pass
        results = ILMLicensedSystemCallSolver._postsolve(self)
        TempfileManager.pop(remove=not self._keepfiles)
        return results