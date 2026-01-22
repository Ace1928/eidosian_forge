from __future__ import unicode_literals
from six import with_metaclass
from collections import defaultdict
import weakref
class SimpleFilter(_FilterType):
    """
    Abstract base class for filters that don't accept any arguments.
    """
    arguments_list = []