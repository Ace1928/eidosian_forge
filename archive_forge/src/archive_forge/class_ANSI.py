import os
from typing import List, Union
class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """
    _bold = '\x1b[1m'
    _gray = '\x1b[90m'
    _red = '\x1b[31m'
    _reset = '\x1b[0m'

    @classmethod
    def bold(cls, s: str) -> str:
        return cls._format(s, cls._bold)

    @classmethod
    def gray(cls, s: str) -> str:
        return cls._format(s, cls._gray)

    @classmethod
    def red(cls, s: str) -> str:
        return cls._format(s, cls._bold + cls._red)

    @classmethod
    def _format(cls, s: str, code: str) -> str:
        if os.environ.get('NO_COLOR'):
            return s
        return f'{code}{s}{cls._reset}'