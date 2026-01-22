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
def ARG(self, node):
    self.addBinding(node, Argument(node.arg, self.getScopeNode(node)))