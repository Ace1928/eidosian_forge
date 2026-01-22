import json
import locale
import multiprocessing
import os
import platform
import textwrap
import sys
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from subprocess import check_output, PIPE, CalledProcessError
import numpy as np
import llvmlite.binding as llvmbind
from llvmlite import __version__ as llvmlite_version
from numba import cuda as cu, __version__ as version_number
from numba.cuda import cudadrv
from numba.cuda.cudadrv.driver import driver as cudriver
from numba.cuda.cudadrv.runtime import runtime as curuntime
from numba.core import config
class CmdBufferOut(tuple):
    buffer_output_flag = True