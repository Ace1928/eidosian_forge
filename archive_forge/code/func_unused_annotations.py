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
def unused_annotations(self):
    """
        Return a generator for the annotations which have not been used.
        """
    for name, binding in self.items():
        if not binding.used and isinstance(binding, Annotation):
            yield (name, binding)