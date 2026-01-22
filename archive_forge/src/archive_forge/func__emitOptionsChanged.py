import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
def _emitOptionsChanged(self, param, data):
    self.emitStateChanged('options', data)