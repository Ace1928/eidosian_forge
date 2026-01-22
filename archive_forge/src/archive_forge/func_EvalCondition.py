import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def EvalCondition(condition, conditions_key, phase, variables, build_file):
    """Returns the dict that should be used or None if the result was
  that nothing should be used."""
    if type(condition) is not list:
        raise GypError(conditions_key + ' must be a list')
    if len(condition) < 2:
        raise GypError(conditions_key + ' ' + condition[0] + ' must be at least length 2, not ' + str(len(condition)))
    i = 0
    result = None
    while i < len(condition):
        cond_expr = condition[i]
        true_dict = condition[i + 1]
        if type(true_dict) is not dict:
            raise GypError('{} {} must be followed by a dictionary, not {}'.format(conditions_key, cond_expr, type(true_dict)))
        if len(condition) > i + 2 and type(condition[i + 2]) is dict:
            false_dict = condition[i + 2]
            i = i + 3
            if i != len(condition):
                raise GypError('{} {} has {} unexpected trailing items'.format(conditions_key, cond_expr, len(condition) - i))
        else:
            false_dict = None
            i = i + 2
        if result is None:
            result = EvalSingleCondition(cond_expr, true_dict, false_dict, phase, variables, build_file)
    return result