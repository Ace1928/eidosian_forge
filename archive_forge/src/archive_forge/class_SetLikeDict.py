from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
class SetLikeDict(dict):
    """a dictionary that has some setlike methods on it"""

    def union(self, other):
        """produce a 'union' of this dict and another (at the key level).

        values in the second dict take precedence over that of the first"""
        x = SetLikeDict(**self)
        x.update(other)
        return x