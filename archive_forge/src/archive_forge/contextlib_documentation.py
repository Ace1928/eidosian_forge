import abc
import os
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias
Helper to correctly register coroutine function to __aexit__
        method.