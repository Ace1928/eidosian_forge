import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_identity(self, node, value):
    return value