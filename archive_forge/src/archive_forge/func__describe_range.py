import os
import stat
import sys
import typing as t
from datetime import datetime
from gettext import gettext as _
from gettext import ngettext
from ._compat import _get_argv_encoding
from ._compat import open_stream
from .exceptions import BadParameter
from .utils import format_filename
from .utils import LazyFile
from .utils import safecall
def _describe_range(self) -> str:
    """Describe the range for use in help text."""
    if self.min is None:
        op = '<' if self.max_open else '<='
        return f'x{op}{self.max}'
    if self.max is None:
        op = '>' if self.min_open else '>='
        return f'x{op}{self.min}'
    lop = '<' if self.min_open else '<='
    rop = '<' if self.max_open else '<='
    return f'{self.min}{lop}x{rop}{self.max}'