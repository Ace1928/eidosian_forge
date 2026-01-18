import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_multi_select_dict(self, node, value):
    if value is None:
        return None
    collected = self._dict_cls()
    for child in node['children']:
        collected[child['value']] = self.visit(child, value)
    return collected