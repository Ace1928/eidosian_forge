import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
class ColorHelpFormatter(_HelpFormatter):

    class _Section(_HelpFormatter._Section):

        @property
        def heading(self) -> Optional[Text]:
            if not self._heading or self._heading == argparse.SUPPRESS or (not getattr(self.formatter, _color_attr, False)):
                return self._heading
            return f'\x1b[4m{self._heading}\x1b[0m'

        @heading.setter
        def heading(self, heading: Optional[Text]) -> None:
            self._heading = heading

    def _metavar_formatter(self, action: argparse.Action, default_metavar: Text) -> Callable[[int], Tuple[str, ...]]:
        get_metavars = super()._metavar_formatter(action, default_metavar)
        if not getattr(self, _color_attr, False):
            return get_metavars

        def color_metavar(size: int) -> Tuple[str, ...]:
            return tuple((f'\x1b[3m{mv}\x1b[0m' for mv in get_metavars(size)))
        return color_metavar