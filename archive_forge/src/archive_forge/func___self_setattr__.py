import sys
import operator
import inspect
def __self_setattr__(self, name, value):
    object.__setattr__(self, name, value)