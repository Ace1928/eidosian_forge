import builtins
import errno
import io
import locale
import os
import time
import signal
import sys
import threading
import warnings
import contextlib
from time import monotonic as _time
import types
class STARTUPINFO:

    def __init__(self, *, dwFlags=0, hStdInput=None, hStdOutput=None, hStdError=None, wShowWindow=0, lpAttributeList=None):
        self.dwFlags = dwFlags
        self.hStdInput = hStdInput
        self.hStdOutput = hStdOutput
        self.hStdError = hStdError
        self.wShowWindow = wShowWindow
        self.lpAttributeList = lpAttributeList or {'handle_list': []}

    def copy(self):
        attr_list = self.lpAttributeList.copy()
        if 'handle_list' in attr_list:
            attr_list['handle_list'] = list(attr_list['handle_list'])
        return STARTUPINFO(dwFlags=self.dwFlags, hStdInput=self.hStdInput, hStdOutput=self.hStdOutput, hStdError=self.hStdError, wShowWindow=self.wShowWindow, lpAttributeList=attr_list)