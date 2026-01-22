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
def _node_matches_argspec(node, func):
    """Returns True is node fits the argspec of func."""
    arg_spec = tf_inspect.getfullargspec(func)
    node_args = tuple((_arg_name(arg) for arg in node.args.args))
    if node_args != tuple(arg_spec.args):
        return False
    if arg_spec.varargs != _arg_name(node.args.vararg):
        return False
    if arg_spec.varkw != _arg_name(node.args.kwarg):
        return False
    node_kwonlyargs = tuple((_arg_name(arg) for arg in node.args.kwonlyargs))
    if node_kwonlyargs != tuple(arg_spec.kwonlyargs):
        return False
    return True