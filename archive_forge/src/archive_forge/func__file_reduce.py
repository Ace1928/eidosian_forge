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
def _file_reduce(obj):
    """Save a file"""
    import io
    if not hasattr(obj, 'name') or not hasattr(obj, 'mode'):
        raise pickle.PicklingError('Cannot pickle files that do not map to an actual file')
    if obj is sys.stdout:
        return (getattr, (sys, 'stdout'))
    if obj is sys.stderr:
        return (getattr, (sys, 'stderr'))
    if obj is sys.stdin:
        raise pickle.PicklingError('Cannot pickle standard input')
    if obj.closed:
        raise pickle.PicklingError('Cannot pickle closed files')
    if hasattr(obj, 'isatty') and obj.isatty():
        raise pickle.PicklingError('Cannot pickle files that map to tty objects')
    if 'r' not in obj.mode and '+' not in obj.mode:
        raise pickle.PicklingError('Cannot pickle files that are not opened for reading: %s' % obj.mode)
    name = obj.name
    retval = io.StringIO()
    try:
        curloc = obj.tell()
        obj.seek(0)
        contents = obj.read()
        obj.seek(curloc)
    except IOError as e:
        raise pickle.PicklingError('Cannot pickle file %s as it cannot be read' % name) from e
    retval.write(contents)
    retval.seek(curloc)
    retval.name = name
    return (_file_reconstructor, (retval,))