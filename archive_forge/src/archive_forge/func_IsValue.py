from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from fire import inspectutils
import six
def IsValue(component):
    return isinstance(component, VALUE_TYPES) or HasCustomStr(component)