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
def _format_exc(self, exc: BaseException):
    return traceback.format_exception_only(exc)[-1].strip()