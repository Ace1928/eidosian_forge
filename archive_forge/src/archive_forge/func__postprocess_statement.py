import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _postprocess_statement(self, node):
    if not self.state[_Block].return_used:
        return (node, None)
    state = self.state[_Block]
    if state.create_guard_now:
        template = '\n        if not do_return_var_name:\n          original_node\n      '
        cond, = templates.replace(template, do_return_var_name=self.state[_Function].do_return_var_name, original_node=node)
        node, block = (cond, cond.body)
    else:
        node, block = (node, None)
    state.create_guard_now = state.create_guard_next
    state.create_guard_next = False
    return (node, block)