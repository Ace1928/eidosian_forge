from __future__ import division, absolute_import
import sys
import warnings
from typing import TYPE_CHECKING, Any, TypeVar, Union, Optional, Dict
from ._version import __version__  # noqa: E402
class _Inf(object):
    """
    An object that is bigger than all other objects.
    """

    def __cmp__(self, other):
        """
        @param other: Another object.
        @type other: any

        @return: 0 if other is inf, 1 otherwise.
        @rtype: C{int}
        """
        if other is _inf:
            return 0
        return 1
    if sys.version_info >= (3,):

        def __lt__(self, other):
            return self.__cmp__(other) < 0

        def __le__(self, other):
            return self.__cmp__(other) <= 0

        def __gt__(self, other):
            return self.__cmp__(other) > 0

        def __ge__(self, other):
            return self.__cmp__(other) >= 0