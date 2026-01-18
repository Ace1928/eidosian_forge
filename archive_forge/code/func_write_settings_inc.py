from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_settings_inc(f, settings, embedded_flag):
    """
    Write prototype for settings structure to file
    """
    f.write('// Settings structure prototype\n')
    f.write('extern OSQPSettings settings;\n\n')