import re
import sys
import types
import copy
import os
import inspect
def _funcs_to_names(funclist, namelist):
    result = []
    for f, name in zip(funclist, namelist):
        if f and f[0]:
            result.append((name, f[1]))
        else:
            result.append(f)
    return result