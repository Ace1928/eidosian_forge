import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
class ProgramNameFilter(logging.Filter):
    """
    Log filter to enable the ``%(programname)s`` format.

    Python's :mod:`logging` module doesn't expose the name of the currently
    running program while I consider this to be a useful addition. Fortunately
    it's very easy to expose additional fields in format strings:
    :func:`filter()` simply sets the ``programname`` attribute of each
    :class:`~logging.LogRecord` object it receives and this is enough to enable
    the use of the ``%(programname)s`` expression in format strings.

    Refer to :class:`HostNameFilter` for an example of how to manually install
    these log filters.
    """

    @classmethod
    def install(cls, handler, fmt, programname=None, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`ProgramNameFilter` (only if needed).

        :param fmt: The log format string to check for ``%(programname)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param programname: Refer to :func:`__init__()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``programname`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'programname'):
                return
        handler.addFilter(cls(programname))

    def __init__(self, programname=None):
        """
        Initialize a :class:`ProgramNameFilter` object.

        :param programname: The program name to use (defaults to the result of
                            :func:`find_program_name()`).
        """
        self.programname = programname or find_program_name()

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `programname` field."""
        record.programname = self.programname
        return 1