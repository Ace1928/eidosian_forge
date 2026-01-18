import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
def visit_multi_select_list(self, node, value):
    if value is None:
        return None
    collected = []
    for child in node['children']:
        collected.append(self.visit(child, value))
    return collected