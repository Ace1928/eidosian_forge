import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def __mro_entries__(self, bases):
    supercls_name = self.__name__

    class Dummy:

        def __init_subclass__(cls):
            subcls_name = cls.__name__
            raise TypeError(f'Cannot subclass an instance of NewType. Perhaps you were looking for: `{subcls_name} = NewType({subcls_name!r}, {supercls_name})`')
    return (Dummy,)