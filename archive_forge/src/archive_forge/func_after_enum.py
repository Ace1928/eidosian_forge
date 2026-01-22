from __future__ import print_function, absolute_import
import sys
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
def after_enum(self):
    ret = self._after_enum
    self._after_enum = False