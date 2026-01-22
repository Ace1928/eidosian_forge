import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def arg_item(attr_name):
    name = attr_name.lstrip('_')
    if name in overrides:
        value = overrides[name]
        if not value and getattr(self, attr_name) is None:
            value = None
    else:
        value = function.resolve(getattr(self, attr_name))
    return (name, value)