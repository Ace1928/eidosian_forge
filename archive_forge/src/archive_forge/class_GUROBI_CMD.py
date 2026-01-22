from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import gurobi_path
import os
import sys
from .. import constants
import warnings
class GUROBI_CMD(LpSolver_CMD):
    """The GUROBI_CMD solver"""
    name = 'GUROBI_CMD'

    def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, gapAbs=None, options=None, warmStart=False, keepFiles=False, path=None, threads=None, logPath=None, mip_start=False):
        """
        :param bool mip: if False, assume LP even if integer variables
        :param bool msg: if False, no log is shown
        :param float timeLimit: maximum time for solver (in seconds)
        :param float gapRel: relative gap tolerance for the solver to stop (in fraction)
        :param float gapAbs: absolute gap tolerance for the solver to stop
        :param int threads: sets the maximum number of threads
        :param list options: list of additional options to pass to solver
        :param bool warmStart: if True, the solver will use the current value of variables as a start
        :param bool keepFiles: if True, files are saved in the current directory and not deleted after solving
        :param str path: path to the solver binary
        :param str logPath: path to the log file
        """
        LpSolver_CMD.__init__(self, gapRel=gapRel, mip=mip, msg=msg, timeLimit=timeLimit, options=options, warmStart=warmStart, path=path, keepFiles=keepFiles, threads=threads, gapAbs=gapAbs, logPath=logPath)

    def defaultPath(self):
        return self.executableExtension('gurobi_cl')

    def available(self):
        """True if the solver is available"""
        if not self.executable(self.path):
            return False
        result = subprocess.Popen(self.path, stdout=subprocess.PIPE, universal_newlines=True)
        out, err = result.communicate()
        if result.returncode == 0:
            return True
        warnings.warn(f'GUROBI error: {out}.')
        return False

    def actualSolve(self, lp):
        """Solve a well formulated lp problem"""
        if not self.executable(self.path):
            raise PulpSolverError('PuLP: cannot execute ' + self.path)
        tmpLp, tmpSol, tmpMst = self.create_tmp_files(lp.name, 'lp', 'sol', 'mst')
        vs = lp.writeLP(tmpLp, writeSOS=1)
        try:
            os.remove(tmpSol)
        except:
            pass
        cmd = self.path
        options = self.options + self.getOptions()
        if self.timeLimit is not None:
            options.append(('TimeLimit', self.timeLimit))
        cmd += ' ' + ' '.join([f'{key}={value}' for key, value in options])
        cmd += f' ResultFile={tmpSol}'
        if self.optionsDict.get('warmStart', False):
            self.writesol(filename=tmpMst, vs=vs)
            cmd += f' InputFile={tmpMst}'
        if lp.isMIP():
            if not self.mip:
                warnings.warn('GUROBI_CMD does not allow a problem to be relaxed')
        cmd += f' {tmpLp}'
        if self.msg:
            pipe = None
        else:
            pipe = open(os.devnull, 'w')
        return_code = subprocess.call(cmd.split(), stdout=pipe, stderr=pipe)
        if pipe is not None:
            pipe.close()
        if return_code != 0:
            raise PulpSolverError('PuLP: Error while trying to execute ' + self.path)
        if not os.path.exists(tmpSol):
            status = constants.LpStatusNotSolved
            values = reducedCosts = shadowPrices = slacks = None
        else:
            status, values, reducedCosts, shadowPrices, slacks = self.readsol(tmpSol)
        self.delete_tmp_files(tmpLp, tmpMst, tmpSol, 'gurobi.log')
        if status != constants.LpStatusInfeasible:
            lp.assignVarsVals(values)
            lp.assignVarsDj(reducedCosts)
            lp.assignConsPi(shadowPrices)
            lp.assignConsSlack(slacks)
        lp.assignStatus(status)
        return status

    def readsol(self, filename):
        """Read a Gurobi solution file"""
        with open(filename) as my_file:
            try:
                next(my_file)
            except StopIteration:
                status = constants.LpStatusNotSolved
                return (status, {}, {}, {}, {})
            status = constants.LpStatusOptimal
            shadowPrices = {}
            slacks = {}
            shadowPrices = {}
            slacks = {}
            values = {}
            reducedCosts = {}
            for line in my_file:
                if line[0] != '#':
                    name, value = line.split()
                    values[name] = float(value)
        return (status, values, reducedCosts, shadowPrices, slacks)

    def writesol(self, filename, vs):
        """Writes a GUROBI solution file"""
        values = [(v.name, v.value()) for v in vs if v.value() is not None]
        rows = []
        for name, value in values:
            rows.append(f'{name} {value}')
        with open(filename, 'w') as f:
            f.write('\n'.join(rows))
        return True

    def getOptions(self):
        params_eq = dict(logPath='LogFile', gapRel='MIPGap', gapAbs='MIPGapAbs', threads='Threads')
        return [(v, self.optionsDict[k]) for k, v in params_eq.items() if k in self.optionsDict and self.optionsDict[k] is not None]