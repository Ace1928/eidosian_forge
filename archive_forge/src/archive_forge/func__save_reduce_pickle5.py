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
def _save_reduce_pickle5(self, func, args, state=None, listitems=None, dictitems=None, state_setter=None, obj=None):
    save = self.save
    write = self.write
    self.save_reduce(func, args, state=None, listitems=listitems, dictitems=dictitems, obj=obj)
    save(state_setter)
    save(obj)
    save(state)
    write(pickle.TUPLE2)
    write(pickle.REDUCE)
    write(pickle.POP)