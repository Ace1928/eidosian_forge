import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
class TooManySplitsError(DocstringError):
    """
    The docstring had too many ---- section splits. Currently we only support
    using up to a single split, to split the docstring into description and
    epilog parts.
    """