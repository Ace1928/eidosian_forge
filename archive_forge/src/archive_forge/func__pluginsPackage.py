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
def _pluginsPackage() -> types.ModuleType:
    import twisted.plugins as package
    return package