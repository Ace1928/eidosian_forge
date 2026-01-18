import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def valueIsDefault(self):
    """Returns True if this parameter's value is equal to the default value."""
    if not self.hasValue() or not self.hasDefault():
        return False
    return fn.eq(self.value(), self.defaultValue())