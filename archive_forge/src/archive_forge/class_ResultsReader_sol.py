import re
from pyomo.opt.base import results
from pyomo.opt.base.formats import ResultsFormat
from pyomo.opt import SolverResults, SolutionStatus, SolverStatus, TerminationCondition
@results.ReaderFactory.register(str(ResultsFormat.sol))
class ResultsReader_sol(results.AbstractResultsReader):
    """
    Class that reads in a *.sol results file and generates a
    SolverResults object.
    """

    def __init__(self, name=None):
        results.AbstractResultsReader.__init__(self, ResultsFormat.sol)
        if not name is None:
            self.name = name

    def __call__(self, filename, res=None, soln=None, suffixes=[]):
        """
        Parse a *.sol file
        """
        try:
            with open(filename, 'r') as f:
                return self._load(f, res, soln, suffixes)
        except ValueError as e:
            with open(filename, 'r') as f:
                fdata = f.read()
            raise ValueError("Error reading '%s': %s.\nSOL File Output:\n%s" % (filename, str(e), fdata))

    def _load(self, fin, res, soln, suffixes):
        if res is None:
            res = SolverResults()
        msg = []
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip()
            if line == 'Options':
                break
            if line:
                msg.append(line)
        msg = '\n'.join(msg)
        z = []
        if line[:7] == 'Options':
            line = fin.readline()
            nopts = int(line)
            need_vbtol = False
            if nopts > 4:
                nopts -= 2
                need_vbtol = True
            for i in range(nopts + 4):
                line = fin.readline()
                z += [int(line)]
            if need_vbtol:
                line = fin.readline()
                z += [float(line)]
        else:
            raise ValueError('no Options line found')
        n = z[nopts + 3]
        m = z[nopts + 1]
        x = []
        y = []
        i = 0
        while i < m:
            line = fin.readline()
            y.append(float(line))
            i += 1
        i = 0
        while i < n:
            line = fin.readline()
            x.append(float(line))
            i += 1
        objno = [0, 0]
        line = fin.readline()
        if line:
            if line[:5] != 'objno':
                raise ValueError("expected 'objno', found '%s'" % line)
            t = line.split()
            if len(t) != 3:
                raise ValueError("expected two numbers in objno line, but found '%s'" % line)
            objno = [int(t[1]), int(t[2])]
        res.solver.message = msg.strip()
        res.solver.message = res.solver.message.replace('\n', '; ')
        if isinstance(res.solver.message, str):
            res.solver.message = res.solver.message.replace(':', '\\x3a')
        res.solver.status = SolverStatus.ok
        soln_status = SolutionStatus.unknown
        if objno[1] >= 0 and objno[1] <= 99:
            objno_message = 'OPTIMAL SOLUTION FOUND!'
            res.solver.termination_condition = TerminationCondition.optimal
            res.solver.status = SolverStatus.ok
            soln_status = SolutionStatus.optimal
        elif objno[1] >= 100 and objno[1] <= 199:
            objno_message = 'Optimal solution indicated, but ERROR LIKELY!'
            res.solver.termination_condition = TerminationCondition.optimal
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.optimal
        elif objno[1] >= 200 and objno[1] <= 299:
            objno_message = 'INFEASIBLE SOLUTION: constraints cannot be satisfied!'
            res.solver.termination_condition = TerminationCondition.infeasible
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.infeasible
        elif objno[1] >= 300 and objno[1] <= 399:
            objno_message = 'UNBOUNDED PROBLEM: the objective can be improved without limit!'
            res.solver.termination_condition = TerminationCondition.unbounded
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.unbounded
        elif objno[1] >= 400 and objno[1] <= 499:
            objno_message = 'EXCEEDED MAXIMUM NUMBER OF ITERATIONS: the solver was stopped by a limit that you set!'
            res.solver.termination_condition = TerminationCondition.maxIterations
            res.solver.status = SolverStatus.warning
            soln_status = SolutionStatus.stoppedByLimit
        elif objno[1] >= 500 and objno[1] <= 599:
            objno_message = 'FAILURE: the solver stopped by an error condition in the solver routines!'
            res.solver.termination_condition = TerminationCondition.internalSolverError
            res.solver.status = SolverStatus.error
            soln_status = SolutionStatus.error
        res.solver.id = objno[1]
        if res.solver.termination_condition in [TerminationCondition.unknown, TerminationCondition.maxIterations, TerminationCondition.minFunctionValue, TerminationCondition.minStepLength, TerminationCondition.globallyOptimal, TerminationCondition.locallyOptimal, TerminationCondition.optimal, TerminationCondition.maxEvaluations, TerminationCondition.other, TerminationCondition.infeasible]:
            if soln is None:
                soln = res.solution.add()
            res.solution.status = soln_status
            soln.status_description = objno_message
            soln.message = msg.strip()
            soln.message = res.solver.message.replace('\n', '; ')
            soln_variable = soln.variable
            i = 0
            for var_value in x:
                soln_variable['v' + str(i)] = {'Value': var_value}
                i = i + 1
            soln_constraint = soln.constraint
            if any((re.match(suf, 'dual') for suf in suffixes)):
                for i in range(0, len(y)):
                    soln_constraint['c' + str(i)] = {'Dual': y[i]}
            line = fin.readline()
            while line:
                line = line.strip()
                if line == '':
                    continue
                line = line.split()
                if line[0] != 'suffix':
                    remaining = ''
                    line = fin.readline()
                    while line:
                        remaining += line.strip() + '; '
                        line = fin.readline()
                    res.solver.message += remaining
                    break
                unmasked_kind = int(line[1])
                kind = unmasked_kind & 3
                convert_function = int
                if unmasked_kind & 4 == 4:
                    convert_function = float
                nvalues = int(line[2])
                tabline = int(line[5])
                suffix_name = fin.readline().strip()
                if any((re.match(suf, suffix_name) for suf in suffixes)):
                    for n in range(tabline):
                        fin.readline()
                    if kind == 0:
                        for cnt in range(nvalues):
                            suf_line = fin.readline().split()
                            key = 'v' + suf_line[0]
                            if key not in soln_variable:
                                soln_variable[key] = {}
                            soln_variable[key][suffix_name] = convert_function(suf_line[1])
                    elif kind == 1:
                        for cnt in range(nvalues):
                            suf_line = fin.readline().split()
                            key = 'c' + suf_line[0]
                            if key not in soln_constraint:
                                soln_constraint[key] = {}
                            translated_suffix_name = suffix_name[0].upper() + suffix_name[1:]
                            soln_constraint[key][translated_suffix_name] = convert_function(suf_line[1])
                    elif kind == 2:
                        for cnt in range(nvalues):
                            suf_line = fin.readline().split()
                            soln.objective.setdefault('o' + suf_line[0], {})[suffix_name] = convert_function(suf_line[1])
                    elif kind == 3:
                        for cnt in range(nvalues):
                            suf_line = fin.readline().split()
                            soln.problem[suffix_name] = convert_function(suf_line[1])
                else:
                    for cnt in range(nvalues):
                        fin.readline()
                line = fin.readline()
        if res.problem.number_of_constraints == 0:
            res.problem.number_of_constraints = m
        res.problem.number_of_variables = n
        res.problem.number_of_objectives = 1
        return res