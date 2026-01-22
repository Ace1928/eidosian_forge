import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def TYPEVAR(self, node):
    self.handleNodeStore(node)
    self.handle_annotation_always_deferred(node.bound, node)