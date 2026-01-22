from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi import settings
from jedi.inference.arguments import TreeArguments
from jedi.inference.value import iterable
from jedi.inference.base_value import NO_VALUES
from jedi.parser_utils import is_scope
def _get_call_string(node):
    if node.parent.type == 'atom_expr':
        return _get_call_string(node.parent)
    code = ''
    leaf = node.get_first_leaf()
    end = node.get_last_leaf().end_pos
    while leaf.start_pos < end:
        code += leaf.value
        leaf = leaf.get_next_leaf()
    return code