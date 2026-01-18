from abc import ABC
import collections
import enum
import functools
import logging
@functools.wraps(behavior)
def wrapped_behavior(*args, **kwargs):
    return _call_logging_exceptions(behavior, message, *args, **kwargs)