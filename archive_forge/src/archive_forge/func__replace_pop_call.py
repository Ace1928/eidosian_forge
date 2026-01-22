import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _replace_pop_call(self, node):
    assert isinstance(node.func, gast.Attribute)
    scope = anno.getanno(node, NodeAnno.ARGS_SCOPE)
    target_node = node.func.value
    if anno.hasanno(target_node, anno.Basic.QN):
        target_name = anno.getanno(target_node, anno.Basic.QN).ssf()
    else:
        target_name = 'list_'
    pop_var_name = self.ctx.namer.new_symbol(target_name, scope.referenced)
    stmt = self.state[_Statement]
    if stmt.pop_uses is None:
        stmt.pop_uses = []
    stmt.pop_uses.append((node, pop_var_name))
    return templates.replace_as_expression('var_name', var_name=pop_var_name)