import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def uses_star_kwargs_in_call(node):
    """Check if an ast.Call node uses arbitrary-length **kwargs.

  This function works with the AST call node format of Python3.5+
  as well as the different AST format of earlier versions of Python.

  Args:
    node: The ast.Call node to check arg values for.

  Returns:
    True if the node uses starred variadic positional args or keyword args.
    False if it does not.
  """
    if sys.version_info[:2] >= (3, 5):
        for keyword in node.keywords:
            if keyword.arg is None:
                return True
    elif node.kwargs:
        return True
    return False