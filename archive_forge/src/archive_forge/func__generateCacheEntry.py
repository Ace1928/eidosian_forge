import os
import pickle
import sys
import types
from typing import Iterable, Optional, Type, TypeVar
from zope.interface import Interface, providedBy
from twisted.python import log
from twisted.python.components import getAdapterFactory
from twisted.python.modules import getModule
from twisted.python.reflect import namedAny
def _generateCacheEntry(provider):
    dropin = CachedDropin(provider.__name__, provider.__doc__)
    for k, v in provider.__dict__.items():
        plugin = IPlugin(v, None)
        if plugin is not None:
            CachedPlugin(dropin, k, v.__doc__, list(providedBy(plugin)))
    return dropin