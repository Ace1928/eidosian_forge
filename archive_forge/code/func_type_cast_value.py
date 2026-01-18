import enum
import errno
import inspect
import os
import sys
import typing as t
from collections import abc
from contextlib import contextmanager
from contextlib import ExitStack
from functools import update_wrapper
from gettext import gettext as _
from gettext import ngettext
from itertools import repeat
from types import TracebackType
from . import types
from .exceptions import Abort
from .exceptions import BadParameter
from .exceptions import ClickException
from .exceptions import Exit
from .exceptions import MissingParameter
from .exceptions import UsageError
from .formatting import HelpFormatter
from .formatting import join_options
from .globals import pop_context
from .globals import push_context
from .parser import _flag_needs_value
from .parser import OptionParser
from .parser import split_opt
from .termui import confirm
from .termui import prompt
from .termui import style
from .utils import _detect_program_name
from .utils import _expand_args
from .utils import echo
from .utils import make_default_short_help
from .utils import make_str
from .utils import PacifyFlushWrapper
def type_cast_value(self, ctx: Context, value: t.Any) -> t.Any:
    """Convert and validate a value against the option's
        :attr:`type`, :attr:`multiple`, and :attr:`nargs`.
        """
    if value is None:
        return () if self.multiple or self.nargs == -1 else None

    def check_iter(value: t.Any) -> t.Iterator[t.Any]:
        try:
            return _check_iter(value)
        except TypeError:
            raise BadParameter(_('Value must be an iterable.'), ctx=ctx, param=self) from None
    if self.nargs == 1 or self.type.is_composite:

        def convert(value: t.Any) -> t.Any:
            return self.type(value, param=self, ctx=ctx)
    elif self.nargs == -1:

        def convert(value: t.Any) -> t.Any:
            return tuple((self.type(x, self, ctx) for x in check_iter(value)))
    else:

        def convert(value: t.Any) -> t.Any:
            value = tuple(check_iter(value))
            if len(value) != self.nargs:
                raise BadParameter(ngettext('Takes {nargs} values but 1 was given.', 'Takes {nargs} values but {len} were given.', len(value)).format(nargs=self.nargs, len=len(value)), ctx=ctx, param=self)
            return tuple((self.type(x, self, ctx) for x in value))
    if self.multiple:
        return tuple((convert(x) for x in check_iter(value)))
    return convert(value)