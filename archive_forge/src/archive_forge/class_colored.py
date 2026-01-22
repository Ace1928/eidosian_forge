from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
class colored:
    """Terminal colored text.

    Example:
        >>> c = colored(enabled=True)
        >>> print(str(c.red('the quick '), c.blue('brown ', c.bold('fox ')),
        ...       c.magenta(c.underline('jumps over')),
        ...       c.yellow(' the lazy '),
        ...       c.green('dog ')))
    """

    def __init__(self, *s: object, **kwargs: Any) -> None:
        self.s: tuple[object, ...] = s
        self.enabled: bool = not IS_WINDOWS and kwargs.get('enabled', True)
        self.op: str = kwargs.get('op', '')
        self.names: dict[str, Any] = {'black': self.black, 'red': self.red, 'green': self.green, 'yellow': self.yellow, 'blue': self.blue, 'magenta': self.magenta, 'cyan': self.cyan, 'white': self.white}

    def _add(self, a: object, b: object) -> str:
        return f'{a}{b}'

    def _fold_no_color(self, a: Any, b: Any) -> str:
        try:
            A = a.no_color()
        except AttributeError:
            A = str(a)
        try:
            B = b.no_color()
        except AttributeError:
            B = str(b)
        return f'{A}{B}'

    def no_color(self) -> str:
        if self.s:
            return str(reduce(self._fold_no_color, self.s))
        return ''

    def embed(self) -> str:
        prefix = ''
        if self.enabled:
            prefix = self.op
        return f'{prefix}{reduce(self._add, self.s)}'

    def __str__(self) -> str:
        suffix = ''
        if self.enabled:
            suffix = RESET_SEQ
        return f'{self.embed()}{suffix}'

    def node(self, s: tuple[object, ...], op: str) -> colored:
        return self.__class__(*s, enabled=self.enabled, op=op)

    def black(self, *s: object) -> colored:
        return self.node(s, fg(30 + BLACK))

    def red(self, *s: object) -> colored:
        return self.node(s, fg(30 + RED))

    def green(self, *s: object) -> colored:
        return self.node(s, fg(30 + GREEN))

    def yellow(self, *s: object) -> colored:
        return self.node(s, fg(30 + YELLOW))

    def blue(self, *s: object) -> colored:
        return self.node(s, fg(30 + BLUE))

    def magenta(self, *s: object) -> colored:
        return self.node(s, fg(30 + MAGENTA))

    def cyan(self, *s: object) -> colored:
        return self.node(s, fg(30 + CYAN))

    def white(self, *s: object) -> colored:
        return self.node(s, fg(30 + WHITE))

    def __repr__(self) -> str:
        return repr(self.no_color())

    def bold(self, *s: object) -> colored:
        return self.node(s, OP_SEQ % 1)

    def underline(self, *s: object) -> colored:
        return self.node(s, OP_SEQ % 4)

    def blink(self, *s: object) -> colored:
        return self.node(s, OP_SEQ % 5)

    def reverse(self, *s: object) -> colored:
        return self.node(s, OP_SEQ % 7)

    def bright(self, *s: object) -> colored:
        return self.node(s, OP_SEQ % 8)

    def ired(self, *s: object) -> colored:
        return self.node(s, fg(40 + RED))

    def igreen(self, *s: object) -> colored:
        return self.node(s, fg(40 + GREEN))

    def iyellow(self, *s: object) -> colored:
        return self.node(s, fg(40 + YELLOW))

    def iblue(self, *s: colored) -> colored:
        return self.node(s, fg(40 + BLUE))

    def imagenta(self, *s: object) -> colored:
        return self.node(s, fg(40 + MAGENTA))

    def icyan(self, *s: object) -> colored:
        return self.node(s, fg(40 + CYAN))

    def iwhite(self, *s: object) -> colored:
        return self.node(s, fg(40 + WHITE))

    def reset(self, *s: object) -> colored:
        return self.node(s or ('',), RESET_SEQ)

    def __add__(self, other: object) -> str:
        return f'{self}{other}'