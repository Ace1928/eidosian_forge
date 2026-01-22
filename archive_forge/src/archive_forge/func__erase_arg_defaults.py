import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
def _erase_arg_defaults(self, node):
    """Erase arg default expressions, which would otherwise be unbound."""
    args = node.args
    for i in range(len(args.defaults)):
        args.defaults[i] = parser.parse_expression('None')
    for i, d in enumerate(args.kw_defaults):
        if d is not None:
            args.kw_defaults[i] = parser.parse_expression('None')
    return node