import inspect
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect
def _process_statement_directive(self, call_node, directive):
    if self.state[_LoopScope].statements_visited > 1:
        raise ValueError('"%s" must be the first statement in the loop block' % directive.__name__)
    if self.state[_LoopScope].level < 2:
        raise ValueError('"%s" must be used inside a statement' % directive.__name__)
    target = self.state[_LoopScope].ast_node
    node_anno = anno.getanno(target, anno.Basic.DIRECTIVES, {})
    node_anno[directive] = _map_args(call_node, directive)
    anno.setanno(target, anno.Basic.DIRECTIVES, node_anno)
    return call_node