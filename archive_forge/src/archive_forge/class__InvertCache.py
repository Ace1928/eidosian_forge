from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from prompt_toolkit.utils import test_callable_args
class _InvertCache(dict):
    """ Cache for inversion operator. """

    def __missing__(self, filter):
        result = _Invert(filter)
        self[filter] = result
        return result