import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
class AttributeDocstrings(ast.NodeVisitor):
    """An ast.NodeVisitor that collects attribute docstrings."""
    attr_docs = None
    prev_attr = None

    def visit(self, node):
        if self.prev_attr and ast_is_literal_str(node):
            attr_name, attr_type, attr_default = self.prev_attr
            self.attr_docs[attr_name] = (ast_get_constant_value(node.value), attr_type, attr_default)
        self.prev_attr = ast_get_attribute(node)
        if isinstance(node, (ast.ClassDef, ast.Module)):
            self.generic_visit(node)

    def get_attr_docs(self, component: T.Any) -> T.Dict[str, T.Tuple[str, T.Optional[str], T.Optional[str]]]:
        """Get attribute docstrings from the given component.

        :param component: component to process (class or module)
        :returns: for each attribute docstring, a tuple with (description,
            type, default)
        """
        self.attr_docs = {}
        self.prev_attr = None
        try:
            source = textwrap.dedent(inspect.getsource(component))
        except OSError:
            pass
        else:
            tree = ast.parse(source)
            if inspect.ismodule(component):
                self.visit(tree)
            elif isinstance(tree, ast.Module) and isinstance(tree.body[0], ast.ClassDef):
                self.visit(tree.body[0])
        return self.attr_docs