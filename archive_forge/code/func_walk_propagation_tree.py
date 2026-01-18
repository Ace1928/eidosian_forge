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
def walk_propagation_tree(logger):
    """
    Walk through the propagation hierarchy of the given logger.

    :param logger: The logger whose hierarchy to walk (a
                   :class:`~logging.Logger` object).
    :returns: A generator of :class:`~logging.Logger` objects.

    .. note:: This uses the undocumented :class:`logging.Logger.parent`
              attribute to find higher level loggers, however it won't
              raise an exception if the attribute isn't available.
    """
    while isinstance(logger, logging.Logger):
        yield logger
        if logger.propagate:
            logger = getattr(logger, 'parent', None)
        else:
            logger = None