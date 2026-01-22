from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
class BooleanOptionalAction(argparse.Action):
    """Adapted from https://github.com/python/cpython/pull/27672"""

    def __init__(self, option_strings: Sequence[str], dest: str, default: _T | str | None=None, type: Callable[[str], _T] | argparse.FileType | None=None, choices: Iterable[_T] | None=None, required: bool=False, help: str | None=None, metavar: str | tuple[str, ...] | None=None) -> None:
        _option_strings = []
        self._no_strings = set()
        for option_string in option_strings:
            _option_strings.append(option_string)
            if option_string.startswith('--'):
                if '.' not in option_string:
                    option_string = '--no' + _strings.get_delimeter() + option_string[2:]
                else:
                    left, _, right = option_string.rpartition('.')
                    option_string = left + '.no' + _strings.get_delimeter() + right
                self._no_strings.add(option_string)
                _option_strings.append(option_string)
        super().__init__(option_strings=_option_strings, dest=dest, nargs=0, default=default, type=type, choices=choices, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            assert option_string is not None
            setattr(namespace, self.dest, option_string not in self._no_strings)

    def format_usage(self):
        return ' | '.join(self.option_strings)