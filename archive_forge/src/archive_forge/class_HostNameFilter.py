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
class HostNameFilter(logging.Filter):
    """
    Log filter to enable the ``%(hostname)s`` format.

    Python's :mod:`logging` module doesn't expose the system's host name while
    I consider this to be a valuable addition. Fortunately it's very easy to
    expose additional fields in format strings: :func:`filter()` simply sets
    the ``hostname`` attribute of each :class:`~logging.LogRecord` object it
    receives and this is enough to enable the use of the ``%(hostname)s``
    expression in format strings.

    You can install this log filter as follows::

     >>> import coloredlogs, logging
     >>> handler = logging.StreamHandler()
     >>> handler.addFilter(coloredlogs.HostNameFilter())
     >>> handler.setFormatter(logging.Formatter('[%(hostname)s] %(message)s'))
     >>> logger = logging.getLogger()
     >>> logger.addHandler(handler)
     >>> logger.setLevel(logging.INFO)
     >>> logger.info("Does it work?")
     [peter-macbook] Does it work?

    Of course :func:`coloredlogs.install()` does all of this for you :-).
    """

    @classmethod
    def install(cls, handler, fmt=None, use_chroot=True, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`HostNameFilter` on a log handler (only if needed).

        :param fmt: The log format string to check for ``%(hostname)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param use_chroot: Refer to :func:`find_hostname()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``hostname`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'hostname'):
                return
        handler.addFilter(cls(use_chroot))

    def __init__(self, use_chroot=True):
        """
        Initialize a :class:`HostNameFilter` object.

        :param use_chroot: Refer to :func:`find_hostname()`.
        """
        self.hostname = find_hostname(use_chroot)

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `hostname` field."""
        record.hostname = self.hostname
        return 1