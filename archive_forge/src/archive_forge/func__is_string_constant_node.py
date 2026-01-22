from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def _is_string_constant_node(node) -> bool:
    return type(node) is ast.Constant and type(node.value) is str