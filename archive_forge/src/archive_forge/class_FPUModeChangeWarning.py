import os
import re
import sys
import numpy as np
import inspect
import sysconfig
class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    pass