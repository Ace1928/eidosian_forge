from collections import defaultdict
from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
class KindMeta(type):
    """
    Metaclass for ``Kind``.

    Assigns empty ``dict`` as class attribute ``_inst`` for every class,
    in order to endow singleton-like behavior.
    """

    def __new__(cls, clsname, bases, dct):
        dct['_inst'] = {}
        return super().__new__(cls, clsname, bases, dct)