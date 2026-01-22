from __future__ import unicode_literals
from six import with_metaclass
from collections import defaultdict
import weakref
class CLIFilter(_FilterType):
    """
    Abstract base class for filters that accept a
    :class:`~prompt_toolkit.interface.CommandLineInterface` argument. It cannot
    be instantiated, it's only to be used for instance assertions, e.g.::

        isinstance(my_filter, CliFilter)
    """
    arguments_list = ['cli']