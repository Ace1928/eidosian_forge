from typing import NamedTuple
from .util import *
from .exceptions import *
from .actions import *
from .core import __diag__, __compat__
from .results import *
from .core import *  # type: ignore[misc, assignment]
from .core import _builtin_exprs as core_builtin_exprs
from .helpers import *  # type: ignore[misc, assignment]
from .helpers import _builtin_exprs as helper_builtin_exprs
from .unicode import unicode_set, UnicodeRangeList, pyparsing_unicode as unicode
from .testing import pyparsing_test as testing
from .common import (
class version_info(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: str
    serial: int

    @property
    def __version__(self):
        return f'{self.major}.{self.minor}.{self.micro}' + (f'{('r' if self.releaselevel[0] == 'c' else '')}{self.releaselevel[0]}{self.serial}', '')[self.releaselevel == 'final']

    def __str__(self):
        return f'{__name__} {self.__version__} / {__version_time__}'

    def __repr__(self):
        return f'{__name__}.{type(self).__name__}({', '.join(('{}={!r}'.format(*nv) for nv in zip(self._fields, self)))})'