import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class ColorRawTextHelpFormatter(ColorHelpFormatter, argparse.RawTextHelpFormatter):
    """Help message formatter which retains formatting of all help text."""