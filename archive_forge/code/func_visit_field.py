import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_field(self, node, value):
    try:
        return value.get(node['value'])
    except AttributeError:
        return None