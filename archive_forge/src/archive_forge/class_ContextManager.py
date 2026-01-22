import re
import sys
import inspect
import operator
import itertools
import collections
from inspect import getfullargspec
class ContextManager(_GeneratorContextManager):

    def __call__(self, func):
        """Context manager decorator"""
        return FunctionMaker.create(func, 'with _self_: return _func_(%(shortsignature)s)', dict(_self_=self, _func_=func), __wrapped__=func)