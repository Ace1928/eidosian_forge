import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def valueModifiedSinceResetToDefault(self):
    """Return True if this parameter's value has been changed since the last time
        it was reset to its default value."""
    return self._modifiedSinceReset