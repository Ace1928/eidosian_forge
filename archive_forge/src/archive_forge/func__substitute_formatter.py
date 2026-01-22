import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def _substitute_formatter(get_fmtr: Callable[[Any], _HelpFormatter]) -> Callable[[argparse.ArgumentParser], _HelpFormatter]:

    @functools.wraps(get_fmtr)
    def _get_formatter(parser: argparse.ArgumentParser) -> _HelpFormatter:
        if parser.formatter_class is _HelpFormatter:
            parser.formatter_class = ColorHelpFormatter
        formatter = get_fmtr(parser)
        if isinstance(formatter, ColorHelpFormatter):
            setattr(formatter, _color_attr, getattr(parser, _color_attr, False))
        return formatter
    return _get_formatter