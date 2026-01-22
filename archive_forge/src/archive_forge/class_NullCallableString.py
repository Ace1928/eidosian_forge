from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
class NullCallableString(text_type):
    """A dummy callable Unicode to stand in for ``FormattingString`` and
    ``ParametrizingString``

    We use this when there is no tty and thus all capabilities should be blank.

    """

    def __new__(cls):
        new = text_type.__new__(cls, u'')
        return new

    def __call__(self, *args):
        """Return a Unicode or whatever you passed in as the first arg
        (hopefully a string of some kind).

        When called with an int as the first arg, return an empty Unicode. An
        int is a good hint that I am a ``ParametrizingString``, as there are
        only about half a dozen string-returning capabilities on OS X's
        terminfo man page which take any param that's not an int, and those are
        seldom if ever used on modern terminal emulators. (Most have to do with
        programming function keys. Blessings' story for supporting
        non-string-returning caps is undeveloped.) And any parametrized
        capability in a situation where all capabilities themselves are taken
        to be blank are, of course, themselves blank.

        When called with a non-int as the first arg (no no args at all), return
        the first arg. I am acting as a ``FormattingString``.

        """
        if len(args) != 1 or isinstance(args[0], int):
            return u''
        return args[0]