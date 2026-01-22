import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def _complete_getattr(user_context, instance):
    """
    A heuristic to make completion for proxy objects work. This is not
    intended to work in all cases. It works exactly in this case:

        def __getattr__(self, name):
            ...
            return getattr(any_object, name)

    It is important that the return contains getattr directly, otherwise it
    won't work anymore. It's really just a stupid heuristic. It will not
    work if you write e.g. `return (getatr(o, name))`, because of the
    additional parentheses. It will also not work if you move the getattr
    to some other place that is not the return statement itself.

    It is intentional that it doesn't work in all cases. Generally it's
    really hard to do even this case (as you can see below). Most people
    will write it like this anyway and the other ones, well they are just
    out of luck I guess :) ~dave.
    """
    names = instance.get_function_slot_names('__getattr__') or instance.get_function_slot_names('__getattribute__')
    functions = ValueSet.from_sets((name.infer() for name in names))
    for func in functions:
        tree_node = func.tree_node
        if tree_node is None or tree_node.type != 'funcdef':
            continue
        for return_stmt in tree_node.iter_return_stmts():
            if return_stmt.type != 'return_stmt':
                continue
            atom_expr = return_stmt.children[1]
            if atom_expr.type != 'atom_expr':
                continue
            atom = atom_expr.children[0]
            trailer = atom_expr.children[1]
            if len(atom_expr.children) != 2 or atom.type != 'name' or atom.value != 'getattr':
                continue
            arglist = trailer.children[1]
            if arglist.type != 'arglist' or len(arglist.children) < 3:
                continue
            context = func.as_context()
            object_node = arglist.children[0]
            name_node = arglist.children[2]
            name_list = context.goto(name_node, name_node.start_pos)
            if not any((n.api_type == 'param' for n in name_list)):
                continue
            objects = context.infer_node(object_node)
            return complete_trailer(user_context, objects)
    return []