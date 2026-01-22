import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _enum_getstate(obj):
    clsdict, slotstate = _class_getstate(obj)
    members = {e.name: e.value for e in obj}
    for attrname in ['_generate_next_value_', '_member_names_', '_member_map_', '_member_type_', '_value2member_map_']:
        clsdict.pop(attrname, None)
    for member in members:
        clsdict.pop(member)
    return (clsdict, slotstate)