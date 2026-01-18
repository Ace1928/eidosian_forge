import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_expref(self, node, value):
    return _Expression(node['children'][0], self)