from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _get_st_write_from_expr(node, i, parent_type, is_root, is_last_expr, file_ends_in_semicolon):
    if type(node.value) is ast.Call and (not _is_displayable_last_expr(is_root, is_last_expr, file_ends_in_semicolon)):
        return None
    if _is_docstring_node(node.value, i, parent_type) and (not _should_display_docstring_like_node_anyway(is_root)):
        return None
    if type(node.value) is ast.Yield or type(node.value) is ast.YieldFrom:
        return None
    if type(node.value) is ast.Await:
        return None
    if type(node.value) is ast.Tuple:
        args = node.value.elts
        st_write = _build_st_write_call(args)
    elif type(node.value) is ast.Str:
        args = [node.value]
        st_write = _build_st_write_call(args)
    elif type(node.value) is ast.Name:
        args = [node.value]
        st_write = _build_st_write_call(args)
    else:
        args = [node.value]
        st_write = _build_st_write_call(args)
    return st_write