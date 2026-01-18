import sys
import contextlib
from datetime import datetime, timedelta
from functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def with_metaclass(meta, *bases):

    class MetaClass(type):

        def __new__(metacls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(MetaClass, 'temporary_class', (), {})