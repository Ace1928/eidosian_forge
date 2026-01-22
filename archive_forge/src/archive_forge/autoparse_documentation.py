import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError

    This context manager allows you to open a filename, if you want to default
    some already-existing file object, like sys.stdout, which shouldn't be
    closed at the end of the context. If the filename argument is a str, bytes,
    or int, the file object is created via a call to open with the given *args
    and **kwargs, sent to the context, and closed at the end of the context,
    just like "with open(filename) as f:". If it isn't one of the openable
    types, the object simply sent to the context unchanged, and left unclosed
    at the end of the context. Example:

        def work_with_file(name=sys.stdout):
            with smart_open(name) as f:
                # Works correctly if name is a str filename or sys.stdout
                print("Some stuff", file=f)
                # If it was a filename, f is closed at the end here.
    