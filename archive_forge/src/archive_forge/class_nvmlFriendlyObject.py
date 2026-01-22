from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class nvmlFriendlyObject(object):

    def __init__(self, dictionary):
        for x in dictionary:
            setattr(self, x, dictionary[x])

    def __str__(self):
        return self.__dict__.__str__()