import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
class KWArgError(AutocommandError):
    """kwarg Error: autocommand can't handle a **kwargs parameter"""