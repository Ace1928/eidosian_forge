from __future__ import annotations
import ast
from typing import Any, Final
from streamlit import config
def add_magic(code: str, script_path: str) -> Any:
    """Modifies the code to support magic Streamlit commands.

    Parameters
    ----------
    code : str
        The Python code.
    script_path : str
        The path to the script file.

    Returns
    -------
    ast.Module
        The syntax tree for the code.

    """
    tree = ast.parse(code, script_path, 'exec')
    file_ends_in_semicolon = _does_file_end_in_semicolon(tree, code)
    return _modify_ast_subtree(tree, is_root=True, file_ends_in_semicolon=file_ends_in_semicolon)