from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _modify_ast_subtree(tree: Any, body_attr: str='body', is_root: bool=False, file_ends_in_semicolon: bool=False):
    """Parses magic commands and modifies the given AST (sub)tree."""
    body = getattr(tree, body_attr)
    for i, node in enumerate(body):
        node_type = type(node)
        if node_type is ast.FunctionDef or node_type is ast.With or node_type is ast.For or (node_type is ast.While) or (node_type is ast.AsyncFunctionDef) or (node_type is ast.AsyncWith) or (node_type is ast.AsyncFor):
            _modify_ast_subtree(node)
        elif node_type is ast.ClassDef:
            for inner_node in node.body:
                if type(inner_node) in {ast.FunctionDef, ast.AsyncFunctionDef}:
                    _modify_ast_subtree(inner_node)
        elif node_type is ast.Try:
            for j, inner_node in enumerate(node.handlers):
                node.handlers[j] = _modify_ast_subtree(inner_node)
            finally_node = _modify_ast_subtree(node, body_attr='finalbody')
            node.finalbody = finally_node.finalbody
            _modify_ast_subtree(node)
        elif node_type is ast.If:
            _modify_ast_subtree(node)
            _modify_ast_subtree(node, 'orelse')
        elif node_type is ast.Expr:
            value = _get_st_write_from_expr(node, i, parent_type=type(tree), is_root=is_root, is_last_expr=i == len(body) - 1, file_ends_in_semicolon=file_ends_in_semicolon)
            if value is not None:
                node.value = value
    if is_root:
        _insert_import_statement(tree)
    ast.fix_missing_locations(tree)
    return tree