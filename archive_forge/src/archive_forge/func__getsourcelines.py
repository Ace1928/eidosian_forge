import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def _getsourcelines(self, obj):
    lines, lineno = inspect.getsourcelines(obj)
    lineno = max(1, lineno)
    return (lines, lineno)