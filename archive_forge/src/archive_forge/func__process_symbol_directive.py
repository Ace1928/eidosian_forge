import inspect
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect
def _process_symbol_directive(self, call_node, directive):
    if len(call_node.args) < 1:
        raise ValueError('"%s" requires a positional first argument as the target' % directive.__name__)
    target = call_node.args[0]
    defs = anno.getanno(target, anno.Static.ORIG_DEFINITIONS)
    for def_ in defs:
        def_.directives[directive] = _map_args(call_node, directive)
    return call_node