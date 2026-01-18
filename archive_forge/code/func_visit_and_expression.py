import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_and_expression(self, node, value):
    matched = self.visit(node['children'][0], value)
    if self._is_false(matched):
        return matched
    return self.visit(node['children'][1], value)