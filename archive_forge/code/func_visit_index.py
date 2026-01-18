import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_index(self, node, value):
    if not isinstance(value, list):
        return None
    try:
        return value[node['value']]
    except IndexError:
        return None