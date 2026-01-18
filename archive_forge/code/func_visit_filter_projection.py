import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_filter_projection(self, node, value):
    base = self.visit(node['children'][0], value)
    if not isinstance(base, list):
        return None
    comparator_node = node['children'][2]
    collected = []
    for element in base:
        if self._is_true(self.visit(comparator_node, element)):
            current = self.visit(node['children'][1], element)
            if current is not None:
                collected.append(current)
    return collected