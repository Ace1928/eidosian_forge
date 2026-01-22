from .core import LpSolver_CMD, subprocess, PulpSolverError
import os
from .. import constants
import warnings
class MIPCL_CMD(LpSolver_CMD):
    """The MIPCL_CMD solver"""
    name = 'MIPCL_CMD'

    def __init__(self, path=None, keepFiles=False, mip=True, msg=True, options=None, timeLimit=None):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param list options: list of additional options to pass to solver
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        """
        LpSolver_CMD.__init__(self, mip=mip, msg=msg, timeLimit=timeLimit, options=options, path=path, keepFiles=keepFiles)

    def defaultPath(self):
        return self.executableExtension('mps_mipcl')

    def available(self):
        """True if the solver is available"""
        return self.executable(self.path)

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpMps, tmpSol = self.create_tmp_files(lp.name, 'mps', 'sol')
        if lp.sense == constants.LpMaximize:
            warnings.warn('MIPCL_CMD does not allow maximization, we will minimize the inverse of the objective function.')
            lp += -lp.objective
        lp.checkDuplicateVars()
        lp.checkLengthVars(52)
        lp.writeMPS(tmpMps, mpsSense=lp.sense)
        try:
            os.remove(tmpSol)
        except:
            pass
        cmd = self.path
        cmd += f' {tmpMps}'
        cmd += f' -solfile {tmpSol}'
        if self.timeLimit is not None:
            cmd += f' -time {self.timeLimit}'
        for option in self.options:
            cmd += ' ' + option
        if lp.isMIP():
            if not self.mip:
                warnings.warn('MIPCL_CMD cannot solve the relaxation of a problem')
        if self.msg:
            pipe = None
        else:
            pipe = open(os.devnull, 'w')
        return_code = subprocess.call(cmd.split(), stdout=pipe, stderr=pipe)
        if lp.sense == constants.LpMaximize:
            lp += -lp.objective
        if return_code != 0:
            raise PulpSolverError('PuLP: Error while trying to execute ' + self.path)
        if not os.path.exists(tmpSol):
            status = constants.LpStatusNotSolved
            status_sol = constants.LpSolutionNoSolutionFound
            values = None
        else:
            status, values, status_sol = self.readsol(tmpSol)
        self.delete_tmp_files(tmpMps, tmpSol)
        lp.assignStatus(status, status_sol)
        if status not in [constants.LpStatusInfeasible, constants.LpStatusNotSolved]:
            lp.assignVarsVals(values)
        return status

    @staticmethod
    def readsol(filename):
        """Read a MIPCL solution file"""
        with open(filename) as f:
            content = f.readlines()
        content = [l.strip() for l in content]
        values = {}
        if not len(content):
            return (constants.LpStatusNotSolved, values, constants.LpSolutionNoSolutionFound)
        first_line = content[0]
        if first_line == '=infeas=':
            return (constants.LpStatusInfeasible, values, constants.LpSolutionInfeasible)
        objective, value = first_line.split()
        if abs(float(value)) >= 99999999950.0:
            return (constants.LpStatusUnbounded, values, constants.LpSolutionUnbounded)
        for line in content[1:]:
            name, value = line.split()
            values[name] = float(value)
        return (constants.LpStatusOptimal, values, constants.LpSolutionIntegerFeasible)