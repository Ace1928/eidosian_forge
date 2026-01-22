import operator
import os
import sys
import warnings
from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError
from .core import scip_path, fscip_path
from .. import constants
from typing import Dict, List, Optional, Tuple
class SCIP_CMD(LpSolver_CMD):
    """The SCIP optimization solver"""
    name = 'SCIP_CMD'

    def __init__(self, path=None, mip=True, keepFiles=False, msg=True, options=None, timeLimit=None, gapRel=None, gapAbs=None, maxNodes=None, logPath=None, threads=None):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int maxNodes: max number of nodes during branching. Stops the solving when reached.
        :param int threads: sets the maximum number of threads
        :param str logPath: path to the log file
        """
        LpSolver_CMD.__init__(self, mip=mip, msg=msg, options=options, path=path, keepFiles=keepFiles, timeLimit=timeLimit, gapRel=gapRel, gapAbs=gapAbs, maxNodes=maxNodes, threads=threads, logPath=logPath)
    SCIP_STATUSES = {'unknown': constants.LpStatusUndefined, 'user interrupt': constants.LpStatusNotSolved, 'node limit reached': constants.LpStatusNotSolved, 'total node limit reached': constants.LpStatusNotSolved, 'stall node limit reached': constants.LpStatusNotSolved, 'time limit reached': constants.LpStatusNotSolved, 'memory limit reached': constants.LpStatusNotSolved, 'gap limit reached': constants.LpStatusOptimal, 'solution limit reached': constants.LpStatusNotSolved, 'solution improvement limit reached': constants.LpStatusNotSolved, 'restart limit reached': constants.LpStatusNotSolved, 'optimal solution found': constants.LpStatusOptimal, 'infeasible': constants.LpStatusInfeasible, 'unbounded': constants.LpStatusUnbounded, 'infeasible or unbounded': constants.LpStatusNotSolved}
    NO_SOLUTION_STATUSES = {constants.LpStatusInfeasible, constants.LpStatusUnbounded, constants.LpStatusNotSolved}

    def defaultPath(self):
        return self.executableExtension(scip_path)

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpLp, tmpSol, tmpOptions = self.create_tmp_files(lp.name, 'lp', 'sol', 'set')
        lp.writeLP(tmpLp)
        file_options: List[str] = []
        if self.timeLimit is not None:
            file_options.append(f'limits/time={self.timeLimit}')
        if 'gapRel' in self.optionsDict:
            file_options.append(f'limits/gap={self.optionsDict['gapRel']}')
        if 'gapAbs' in self.optionsDict:
            file_options.append(f'limits/absgap={self.optionsDict['gapAbs']}')
        if 'maxNodes' in self.optionsDict:
            file_options.append(f'limits/nodes={self.optionsDict['maxNodes']}')
        if 'threads' in self.optionsDict and int(self.optionsDict['threads']) > 1:
            warnings.warn('SCIP can only run with a single thread - use FSCIP_CMD for a parallel version of SCIP')
        if not self.mip:
            warnings.warn(f'{self.name} does not allow a problem to be relaxed')
        command: List[str] = []
        command.append(self.path)
        command.extend(['-s', tmpOptions])
        if not self.msg:
            command.append('-q')
        if 'logPath' in self.optionsDict:
            command.extend(['-l', self.optionsDict['logPath']])
        options = iter(self.options)
        for option in options:
            if option.startswith('-'):
                argument = next(options)
                command.extend([option, argument])
            else:
                if '=' not in option:
                    argument = next(options)
                    option += f'={argument}'
                file_options.append(option)
        command.extend(['-c', f'read "{tmpLp}"'])
        command.extend(['-c', 'optimize'])
        command.extend(['-c', f'write solution "{tmpSol}"'])
        command.extend(['-c', 'quit'])
        with open(tmpOptions, 'w') as options_file:
            options_file.write('\n'.join(file_options))
        subprocess.check_call(command, stdout=sys.stdout, stderr=sys.stderr)
        if not os.path.exists(tmpSol):
            raise PulpSolverError('PuLP: Error while executing ' + self.path)
        status, values = self.readsol(tmpSol)
        finalVals = {}
        for v in lp.variables():
            finalVals[v.name] = values.get(v.name, 0.0)
        lp.assignVarsVals(finalVals)
        lp.assignStatus(status)
        self.delete_tmp_files(tmpLp, tmpSol, tmpOptions)
        return status

    @staticmethod
    def readsol(filename):
        """Read a SCIP solution file"""
        with open(filename) as f:
            try:
                line = f.readline()
                comps = line.split(': ')
                assert comps[0] == 'solution status'
                assert len(comps) == 2
            except Exception:
                raise PulpSolverError(f"Can't get SCIP solver status: {line!r}")
            status = SCIP_CMD.SCIP_STATUSES.get(comps[1].strip(), constants.LpStatusUndefined)
            values = {}
            if status in SCIP_CMD.NO_SOLUTION_STATUSES:
                return (status, values)
            try:
                line = f.readline()
                comps = line.split(': ')
                assert comps[0] == 'objective value'
                assert len(comps) == 2
                float(comps[1].strip())
            except Exception:
                raise PulpSolverError(f"Can't get SCIP solver objective: {line!r}")
            for line in f:
                try:
                    comps = line.split()
                    values[comps[0]] = float(comps[1])
                except:
                    raise PulpSolverError(f"Can't read SCIP solver output: {line!r}")
            return (status, values)