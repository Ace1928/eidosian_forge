import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def which_gdb(path_or_bin):
    gdb = shutil.which(path_or_bin)
    return gdb if gdb is not None else path_or_bin