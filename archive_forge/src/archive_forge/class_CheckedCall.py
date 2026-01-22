import contextlib
import ctypes
from ctypes import (
import os
import platform
from shutil import which as _executable_exists
import subprocess
import time
import warnings
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
class CheckedCall:

    def __init__(self, f) -> None:
        super().__setattr__('f', f)

    def __call__(self, *args):
        ret = self.f(*args)
        if not ret and get_errno():
            raise PyperclipWindowsException('Error calling ' + self.f.__name__)
        return ret

    def __setattr__(self, key, value):
        setattr(self.f, key, value)