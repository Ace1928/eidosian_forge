import os
import sys
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.core.kernel.block import IBlock
from pyomo.core import ConcreteModel, Var, Objective
from .gurobi_direct import gurobipy_available
from .ASL import ASL
@SolverFactory.register('_gurobi_shell', doc='Shell interface to the GUROBI LP/MIP solver')
class GUROBISHELL(ILMLicensedSystemCallSolver):
    """Shell interface to the GUROBI LP/MIP solver"""
    _solver_info_cache = {}

    def __init__(self, **kwds):
        kwds['type'] = 'gurobi'
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

    def license_is_valid(self):
        """
        Runs a check for a valid Gurobi license using the
        given executable (default is 'gurobi_cl'). All
        output is hidden. If the test fails for any reason
        (including the executable being invalid), then this
        function will return False.
        """
        solver_exec = self.executable()
        if (solver_exec, 'licensed') in self._solver_info_cache:
            return self._solver_info_cache[solver_exec, 'licensed']
        if not solver_exec:
            licensed = False
        else:
            executable = os.path.join(os.path.dirname(solver_exec), 'gurobi_cl')
            try:
                rc = subprocess.call([executable, '--license'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError:
                try:
                    rc = subprocess.run([solver_exec], input='import gurobipy; gurobipy.Env().dispose(); quit()', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).returncode
                except OSError:
                    rc = 1
            licensed = not rc
        self._solver_info_cache[solver_exec, 'licensed'] = licensed
        return licensed

    def _default_results_format(self, prob_format):
        return ResultsFormat.soln

    def warm_start_capable(self):
        return True

    def _warm_start(self, instance):
        from pyomo.core.base import Var
        output_index = 0
        if isinstance(instance, IBlock):
            smap = getattr(instance, '._symbol_maps')[self._smap_id]
        else:
            smap = instance.solutions.symbol_map[self._smap_id]
        byObject = smap.byObject
        with open(self._warm_start_file_name, 'w') as mst_file:
            for vdata in instance.component_data_objects(Var):
                if vdata.value is not None and id(vdata) in byObject:
                    name = byObject[id(vdata)]
                    mst_file.write('%s %s\n' % (name, vdata.value))

    def _presolve(self, *args, **kwds):
        TempfileManager.push()
        self._warm_start_solve = kwds.pop('warmstart', False)
        self._warm_start_file_name = kwds.pop('warmstart_file', None)
        user_warmstart = False
        if self._warm_start_file_name is not None:
            user_warmstart = True
        if self._warm_start_solve and isinstance(args[0], str):
            pass
        elif self._warm_start_solve and (not isinstance(args[0], str)):
            if self._warm_start_file_name is None:
                assert not user_warmstart
                self._warm_start_file_name = TempfileManager.create_tempfile(suffix='.gurobi.mst')
        ILMLicensedSystemCallSolver._presolve(self, *args, **kwds)
        if len(args) > 0 and (not isinstance(args[0], str)):
            if len(args) != 1:
                raise ValueError('GUROBI _presolve method can only handle a single problem instance - %s were supplied' % (len(args),))
            if self._warm_start_solve and (not user_warmstart):
                start_time = time.time()
                self._warm_start(args[0])
                end_time = time.time()
                if self._report_timing is True:
                    print('Warm start write time=%.2f seconds' % (end_time - start_time))

    def _default_executable(self):
        if sys.platform == 'win32':
            executable = Executable('gurobi.bat')
        else:
            executable = Executable('gurobi.sh')
        if executable:
            return executable.path()
        if gurobipy_available:
            return sys.executable
        logger.warning("Could not locate the 'gurobi' executable, which is required for solver %s" % self.name)
        self.enable = False
        return None

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        solver_exec = self.executable()
        if (solver_exec, 'version') in self._solver_info_cache:
            return self._solver_info_cache[solver_exec, 'version']
        if solver_exec is None:
            ver = _extract_version('')
        else:
            results = subprocess.run([solver_exec], input='import gurobipy; print(gurobipy.gurobi.version()); quit()', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            ver = None
            try:
                ver = tuple(eval(results.stdout.strip()))
                while len(ver) < 4:
                    ver += (0,)
            except SyntaxError:
                ver = _extract_version('')
        if ver is not None:
            ver = ver[:4]
        self._solver_info_cache[solver_exec, 'version'] = ver
        return ver

    def create_command_line(self, executable, problem_files):
        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.gurobi.log')
        if self._soln_file is None:
            self._soln_file = TempfileManager.create_tempfile(suffix='.gurobi.txt')
        problem_filename = self._problem_files[0]
        solution_filename = self._soln_file
        warmstart_filename = self._warm_start_file_name
        options_dict = {}
        for key in self.options:
            options_dict[key] = self.options[key]
        script = 'import sys\n'
        script += 'from gurobipy import *\n'
        script += 'sys.path.append(%r)\n' % (this_file_dir(),)
        script += 'from GUROBI_RUN import *\n'
        script += 'gurobi_run('
        mipgap = float(self.options.mipgap) if self.options.mipgap is not None else None
        for x in (problem_filename, warmstart_filename, solution_filename, None, options_dict, self._suffixes):
            script += '%r,' % x
        script += ')\n'
        script += 'quit()\n'
        if self._keepfiles:
            script_fname = TempfileManager.create_tempfile(suffix='.gurobi.script')
            script_file = open(script_fname, 'w')
            script_file.write(script)
            script_file.close()
            print("Solver script file: '%s'" % script_fname)
            if self._warm_start_solve and self._warm_start_file_name is not None:
                print('Solver warm-start file: ' + self._warm_start_file_name)
        cmd = [executable]
        if self._timer:
            cmd.insert(0, self._timer)
        return Bunch(cmd=cmd, script=script, log_file=self._log_file, env=None)

    def process_soln_file(self, results):
        extract_duals = False
        extract_slacks = False
        extract_rc = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, 'dual'):
                extract_duals = True
                flag = True
            if re.match(suffix, 'slack'):
                extract_slacks = True
                flag = True
            if re.match(suffix, 'rc'):
                extract_rc = True
                flag = True
            if not flag:
                raise RuntimeError('***The GUROBI solver plugin cannot extract solution suffix=' + suffix)
        if not os.path.exists(self._soln_file):
            return
        soln = Solution()
        soln_variables = soln.variable
        soln_constraints = soln.constraint
        num_variables_read = 0
        section = 0
        solution_seen = False
        range_duals = {}
        range_slacks = {}
        INPUT = open(self._soln_file, 'r')
        for line in INPUT:
            line = line.strip()
            tokens = [token.strip() for token in line.split(':')]
            if tokens[0] == 'section':
                if tokens[1] == 'problem':
                    section = 1
                elif tokens[1] == 'solution':
                    section = 2
                    solution_seen = True
                elif tokens[1] == 'solver':
                    section = 3
            elif section == 2:
                if tokens[0] == 'var':
                    if tokens[1] != 'ONE_VAR_CONSTANT':
                        soln_variables[tokens[1]] = {'Value': float(tokens[2])}
                        num_variables_read += 1
                elif tokens[0] == 'status':
                    soln.status = getattr(SolutionStatus, tokens[1])
                elif tokens[0] == 'gap':
                    soln.gap = float(tokens[1])
                elif tokens[0] == 'objective':
                    if tokens[1].strip() != 'None':
                        soln.objective['__default_objective__'] = {'Value': float(tokens[1])}
                        if results.problem.sense == ProblemSense.minimize:
                            results.problem.upper_bound = float(tokens[1])
                        else:
                            results.problem.lower_bound = float(tokens[1])
                elif tokens[0] == 'constraintdual':
                    name = tokens[1]
                    if name != 'c_e_ONE_VAR_CONSTANT':
                        if name.startswith('c_'):
                            soln_constraints.setdefault(tokens[1], {})['Dual'] = float(tokens[2])
                        elif name.startswith('r_l_'):
                            range_duals.setdefault(name[4:], [0, 0])[0] = float(tokens[2])
                        elif name.startswith('r_u_'):
                            range_duals.setdefault(name[4:], [0, 0])[1] = float(tokens[2])
                elif tokens[0] == 'constraintslack':
                    name = tokens[1]
                    if name != 'c_e_ONE_VAR_CONSTANT':
                        if name.startswith('c_'):
                            soln_constraints.setdefault(tokens[1], {})['Slack'] = float(tokens[2])
                        elif name.startswith('r_l_'):
                            range_slacks.setdefault(name[4:], [0, 0])[0] = float(tokens[2])
                        elif name.startswith('r_u_'):
                            range_slacks.setdefault(name[4:], [0, 0])[1] = float(tokens[2])
                elif tokens[0] == 'varrc':
                    if tokens[1] != 'ONE_VAR_CONSTANT':
                        soln_variables[tokens[1]]['Rc'] = float(tokens[2])
                else:
                    setattr(soln, tokens[0], tokens[1])
            elif section == 1:
                if tokens[0] == 'sense':
                    if tokens[1] == 'minimize':
                        results.problem.sense = ProblemSense.minimize
                    elif tokens[1] == 'maximize':
                        results.problem.sense = ProblemSense.maximize
                else:
                    try:
                        val = eval(tokens[1])
                    except:
                        val = tokens[1]
                    setattr(results.problem, tokens[0], val)
            elif section == 3:
                if tokens[0] == 'status':
                    results.solver.status = getattr(SolverStatus, tokens[1])
                elif tokens[0] == 'termination_condition':
                    try:
                        results.solver.termination_condition = getattr(TerminationCondition, tokens[1])
                    except AttributeError:
                        results.solver.termination_condition = TerminationCondition.unknown
                else:
                    setattr(results.solver, tokens[0], tokens[1])
        INPUT.close()
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
        if solution_seen:
            results.solution.insert(soln)

    def _postsolve(self):
        filename_list = os.listdir('.')
        for filename in filename_list:
            try:
                if re.match('gurobi\\.log', filename) != None:
                    os.remove(filename)
            except OSError:
                pass
        results = ILMLicensedSystemCallSolver._postsolve(self)
        TempfileManager.pop(remove=not self._keepfiles)
        return results