import linecache
import sys
import time
import types
from importlib import reload
from types import ModuleType
from typing import Dict
from twisted.python import log, reflect
def updateInstance(self):
    """
    Updates an instance to be current.
    """
    self.__class__ = latestClass(self.__class__)