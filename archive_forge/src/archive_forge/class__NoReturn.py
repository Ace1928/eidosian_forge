import abc
import collections
import collections.abc
import operator
import sys
import typing
class _NoReturn(typing._FinalTypingBase, _root=True):
    """Special type indicating functions that never return.
        Example::

          from typing import NoReturn

          def stop() -> NoReturn:
              raise Exception('no way')

        This type is invalid in other positions, e.g., ``List[NoReturn]``
        will fail in static type checkers.
        """
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError('NoReturn cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError('NoReturn cannot be used with issubclass().')