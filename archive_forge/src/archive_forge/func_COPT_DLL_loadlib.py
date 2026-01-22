import os
import sys
import ctypes
import subprocess
import warnings
from uuid import uuid4
from .core import sparse, ctypesArrayFill, PulpSolverError
from .core import clock, log
from .core import LpSolver, LpSolver_CMD
from ..constants import (
from ..constants import LpContinuous, LpBinary, LpInteger
from ..constants import LpConstraintEQ, LpConstraintLE, LpConstraintGE
from ..constants import LpMinimize, LpMaximize
def COPT_DLL_loadlib():
    """
    Load COPT shared library in all supported platforms
    """
    from glob import glob
    libfile = None
    libpath = None
    libhome = os.getenv('COPT_HOME')
    if sys.platform == 'win32':
        libfile = glob(os.path.join(libhome, 'bin', 'copt.dll'))
    elif sys.platform == 'linux':
        libfile = glob(os.path.join(libhome, 'lib', 'libcopt.so'))
    elif sys.platform == 'darwin':
        libfile = glob(os.path.join(libhome, 'lib', 'libcopt.dylib'))
    else:
        raise PulpSolverError('COPT_PULP: Unsupported operating system')
    if libfile:
        libpath = libfile[0]
    if libpath is None:
        raise PulpSolverError('COPT_PULP: Failed to locate solver library, please refer to COPT manual for installation guide')
    elif sys.platform == 'win32':
        coptlib = ctypes.windll.LoadLibrary(libpath)
    else:
        coptlib = ctypes.cdll.LoadLibrary(libpath)
    return coptlib