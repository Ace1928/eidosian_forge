import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
@register_error
class NotImplementedForSymbol(MXNetError):
    """Error: Not implemented for symbol"""

    def __init__(self, function, alias, *args):
        super(NotImplementedForSymbol, self).__init__()
        self.function = function.__name__
        self.alias = alias
        self.args = [str(type(a)) for a in args]

    def __str__(self):
        msg = 'Function {}'.format(self.function)
        if self.alias:
            msg += ' (namely operator "{}")'.format(self.alias)
        if self.args:
            msg += ' with arguments ({})'.format(', '.join(self.args))
        msg += ' is not implemented for Symbol and only available in NDArray.'
        return msg