from __future__ import annotations
from ruamel.yaml.error import MarkedYAMLError, CommentMark  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.docinfo import Version, Tag  # NOQA
from ruamel.yaml.compat import check_anchorname_char, _debug, nprint, nprintf  # NOQA
class CommentBase:
    __slots__ = ('value', 'line', 'column', 'used', 'function', 'fline', 'ufun', 'uline')

    def __init__(self, value: Any, line: Any, column: Any) -> None:
        self.value = value
        self.line = line
        self.column = column
        self.used = ' '
        if _debug != 0:
            import inspect
            info = inspect.getframeinfo(inspect.stack()[3][0])
            self.function = info.function
            self.fline = info.lineno
            self.ufun = None
            self.uline = None

    def set_used(self, v: Any='+') -> None:
        self.used = v
        if _debug != 0:
            import inspect
            info = inspect.getframeinfo(inspect.stack()[1][0])
            self.ufun = info.function
            self.uline = info.lineno

    def set_assigned(self) -> None:
        self.used = '|'

    def __str__(self) -> str:
        return f'{self.value}'

    def __repr__(self) -> str:
        return f'{self.value!r}'

    def info(self) -> str:
        xv = self.value + '"'
        name = self.name
        return f'{name}{self.used} {self.line:2}:{self.column:<2} "{xv:40s} {self.function}:{self.fline} {self.ufun}:{self.uline}'