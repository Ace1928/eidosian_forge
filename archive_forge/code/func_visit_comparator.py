import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_comparator(self, node, value):
    comparator_func = self.COMPARATOR_FUNC[node['value']]
    if node['value'] in self._EQUALITY_OPS:
        return comparator_func(self.visit(node['children'][0], value), self.visit(node['children'][1], value))
    else:
        left = self.visit(node['children'][0], value)
        right = self.visit(node['children'][1], value)
        num_types = (int, float)
        if not (_is_comparable(left) and _is_comparable(right)):
            return None
        return comparator_func(left, right)