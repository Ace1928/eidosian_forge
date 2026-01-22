import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
class QueueMethod:
    """
    I represent a method that doesn't exist yet.
    """

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __call__(self, *args):
        self.calls.append((self.name, args))