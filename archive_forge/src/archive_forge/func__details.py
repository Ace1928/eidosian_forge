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
@functools.cached_property
def _details(self):
    import runpy
    return runpy._get_module_details(self)