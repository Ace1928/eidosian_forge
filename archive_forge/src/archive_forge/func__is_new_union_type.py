import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def _is_new_union_type(instance: Any) -> bool:
    new_union_type = getattr(types, 'UnionType', None)
    return new_union_type is not None and isinstance(instance, new_union_type)