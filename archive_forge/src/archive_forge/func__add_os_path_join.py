import os
from jedi.api import classes
from jedi.api.strings import StringName, get_quote_ending
from jedi.api.helpers import match
from jedi.inference.helpers import get_str_or_none
def _add_os_path_join(module_context, start_leaf, bracket_start):

    def check(maybe_bracket, nodes):
        if maybe_bracket.start_pos != bracket_start:
            return None
        if not nodes:
            return ''
        context = module_context.create_context(nodes[0])
        return _add_strings(context, nodes, add_slash=True) or ''
    if start_leaf.type == 'error_leaf':
        value_node = start_leaf.parent
        index = value_node.children.index(start_leaf)
        if index > 0:
            error_node = value_node.children[index - 1]
            if error_node.type == 'error_node' and len(error_node.children) >= 2:
                index = -2
                if error_node.children[-1].type == 'arglist':
                    arglist_nodes = error_node.children[-1].children
                    index -= 1
                else:
                    arglist_nodes = []
                return check(error_node.children[index + 1], arglist_nodes[::2])
        return None
    searched_node_child = start_leaf
    while searched_node_child.parent is not None and searched_node_child.parent.type not in ('arglist', 'trailer', 'error_node'):
        searched_node_child = searched_node_child.parent
    if searched_node_child.get_first_leaf() is not start_leaf:
        return None
    searched_node = searched_node_child.parent
    if searched_node is None:
        return None
    index = searched_node.children.index(searched_node_child)
    arglist_nodes = searched_node.children[:index]
    if searched_node.type == 'arglist':
        trailer = searched_node.parent
        if trailer.type == 'error_node':
            trailer_index = trailer.children.index(searched_node)
            assert trailer_index >= 2
            assert trailer.children[trailer_index - 1] == '('
            return check(trailer.children[trailer_index - 1], arglist_nodes[::2])
        elif trailer.type == 'trailer':
            return check(trailer.children[0], arglist_nodes[::2])
    elif searched_node.type == 'trailer':
        return check(searched_node.children[0], [])
    elif searched_node.type == 'error_node':
        return check(arglist_nodes[-1], [])