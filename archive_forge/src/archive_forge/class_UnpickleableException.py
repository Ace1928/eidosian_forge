import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
class UnpickleableException(Exception):
    """
    An exception representing another Exception that could not be pickled.
    """

    @staticmethod
    def dump(type, exc):
        """
        Always return a dumped (pickled) type and exc. If exc can't be pickled,
        wrap it in UnpickleableException first.
        """
        try:
            return (pickle.dumps(type), pickle.dumps(exc))
        except Exception:
            from setuptools.sandbox import UnpickleableException as cls
            return cls.dump(cls, cls(repr(exc)))