from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
def _to_callables(context, trailer):
    from jedi.inference.syntax_tree import infer_trailer
    atom_expr = trailer.parent
    index = atom_expr.children[0] == 'await'
    values = context.infer_node(atom_expr.children[index])
    for trailer2 in atom_expr.children[index + 1:]:
        if trailer == trailer2:
            break
        values = infer_trailer(context, values, trailer2)
    return values