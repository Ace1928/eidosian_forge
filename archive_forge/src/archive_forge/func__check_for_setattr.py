from parso.python import tree
from jedi import debug
from jedi.inference.helpers import is_string
def _check_for_setattr(instance):
    """
    Check if there's any setattr method inside an instance. If so, return True.
    """
    module = instance.get_root_context()
    node = module.tree_node
    if node is None:
        return False
    try:
        stmt_names = node.get_used_names()['setattr']
    except KeyError:
        return False
    return any((node.start_pos < n.start_pos < node.end_pos and (not (n.parent.type == 'funcdef' and n.parent.name == n)) for n in stmt_names))