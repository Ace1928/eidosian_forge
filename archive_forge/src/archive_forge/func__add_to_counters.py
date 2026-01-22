from collections.abc import Container, Iterable, Sized, Hashable
from functools import reduce
from typing import Generic, TypeVar
from pyrsistent._pmap import pmap
def _add_to_counters(counters, element):
    return counters.set(element, counters.get(element, 0) + 1)