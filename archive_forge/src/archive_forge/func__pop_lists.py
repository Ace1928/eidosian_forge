from collections.abc import Sequence, Hashable
from itertools import islice, chain
from numbers import Integral
from typing import TypeVar, Generic
from pyrsistent._plist import plist
@staticmethod
def _pop_lists(primary_list, secondary_list, count):
    new_primary_list = primary_list
    new_secondary_list = secondary_list
    while count > 0 and (new_primary_list or new_secondary_list):
        count -= 1
        if new_primary_list.rest:
            new_primary_list = new_primary_list.rest
        elif new_primary_list:
            new_primary_list = new_secondary_list.reverse()
            new_secondary_list = plist()
        else:
            new_primary_list = new_secondary_list.reverse().rest
            new_secondary_list = plist()
    return (new_primary_list, new_secondary_list)