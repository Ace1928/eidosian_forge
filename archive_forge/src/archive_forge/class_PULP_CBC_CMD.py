from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
class PULP_CBC_CMD(COIN_CMD):
    """
    This solver uses a precompiled version of cbc provided with the package
    """
    name = 'PULP_CBC_CMD'
    pulp_cbc_path = pulp_cbc_path
    try:
        if os.name != 'nt':
            if not os.access(pulp_cbc_path, os.X_OK):
                import stat
                os.chmod(pulp_cbc_path, stat.S_IXUSR + stat.S_IXOTH)
    except:

        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp, callback=None):
            """Solve a well formulated lp problem"""
            raise PulpSolverError('PULP_CBC_CMD: Not Available (check permissions on %s)' % self.pulp_cbc_path)
    else:

        def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, gapAbs=None, presolve=None, cuts=None, strong=None, options=None, warmStart=False, keepFiles=False, path=None, threads=None, logPath=None, timeMode='elapsed'):
            if path is not None:
                raise PulpSolverError('Use COIN_CMD if you want to set a path')
            COIN_CMD.__init__(self, path=self.pulp_cbc_path, mip=mip, msg=msg, timeLimit=timeLimit, gapRel=gapRel, gapAbs=gapAbs, presolve=presolve, cuts=cuts, strong=strong, options=options, warmStart=warmStart, keepFiles=keepFiles, threads=threads, logPath=logPath, timeMode=timeMode)