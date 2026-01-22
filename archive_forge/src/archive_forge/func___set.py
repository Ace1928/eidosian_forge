import re
import string
import types
def __set(self, key, real_value, coded_value):
    """Private method for setting a cookie's value"""
    M = self.get(key, Morsel())
    M.set(key, real_value, coded_value)
    dict.__setitem__(self, key, M)