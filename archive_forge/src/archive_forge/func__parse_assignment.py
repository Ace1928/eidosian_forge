import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _parse_assignment(node: ast.Assign) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Parse an assignment and return a tuple of name, value."""
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        var_name = node.targets[0].id
        return (var_name, node.value)
    return (None, None)