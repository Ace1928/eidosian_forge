import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
class _StringIO(object):
    """Alternative to built-in StringIO needed to circumvent unicode/ascii issues"""

    def __init__(self):
        self.data = []

    def write(self, data):
        self.data.append(data)

    def getvalue(self):
        return ''.join(map(str, self.data)).encode('utf8')