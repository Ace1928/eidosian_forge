import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
def _assignment(ast):
    """Takes the AST representation of an assignment, and returns a
    function that applies the assignment of a given value to a dictionary.
    """

    def _names(node):
        if isinstance(node, _ast.Tuple):
            return tuple([_names(child) for child in node.elts])
        elif isinstance(node, _ast.Name):
            return node.id

    def _assign(data, value, names=_names(ast)):
        if type(names) is tuple:
            for idx in range(len(names)):
                _assign(data, value[idx], names[idx])
        else:
            data[names] = value
    return _assign