import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
def _arg_name(node):
    if node is None:
        return None
    if isinstance(node, gast.Name):
        return node.id
    assert isinstance(node, str)
    return node