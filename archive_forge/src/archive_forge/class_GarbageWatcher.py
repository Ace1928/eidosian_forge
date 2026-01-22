from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
class GarbageWatcher(object):
    """
    Convenient dictionary for holding weak references to objects.
    Mainly used to check whether the objects have been collect yet or not.
    
    Example:
        gw = GarbageWatcher()
        gw['objName'] = obj
        gw['objName2'] = obj2
        gw.check()  
        
    
    """

    def __init__(self):
        self.objs = weakref.WeakValueDictionary()
        self.allNames = []

    def add(self, obj, name):
        self.objs[name] = obj
        self.allNames.append(name)

    def __setitem__(self, name, obj):
        self.add(obj, name)

    def check(self):
        """Print a list of all watched objects and whether they have been collected."""
        gc.collect()
        dead = self.allNames[:]
        alive = []
        for k in self.objs:
            dead.remove(k)
            alive.append(k)
        print('Deleted objects:', dead)
        print('Live objects:', alive)

    def __getitem__(self, item):
        return self.objs[item]