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
def _validate_provider_return_type(function: Callable, return_type: type, allow_multi: bool) -> None:
    origin = _get_origin(_punch_through_alias(return_type))
    if origin in {dict, list} and (not allow_multi):
        raise Error('Function %s needs to be decorated with multiprovider instead of provider if it is to provide values to a multibinding of type %s' % (function.__name__, return_type))