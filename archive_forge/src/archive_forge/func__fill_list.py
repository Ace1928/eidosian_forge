from abc import abstractmethod, ABCMeta
from collections.abc import Sequence, Hashable
from numbers import Integral
import operator
from typing import TypeVar, Generic
from pyrsistent._transformations import transform
def _fill_list(self, node, shift, the_list):
    if shift:
        shift -= SHIFT
        for n in node:
            self._fill_list(n, shift, the_list)
    else:
        the_list.extend(node)