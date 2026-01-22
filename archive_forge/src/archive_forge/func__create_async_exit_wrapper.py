import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
@staticmethod
def _create_async_exit_wrapper(cm, cm_exit):
    return MethodType(cm_exit, cm)