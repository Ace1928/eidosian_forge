import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _replace_stack_call(self, node):
    assert len(node.args) == 1
    dtype = self.get_definition_directive(node.args[0], directives.set_element_type, 'dtype', default=templates.replace_as_expression('None'))
    template = '\n      ag__.list_stack(\n          target,\n          opts=ag__.ListStackOpts(\n              element_dtype=dtype,\n              original_call=orig_call))\n    '
    return templates.replace_as_expression(template, dtype=dtype, target=node.args[0], orig_call=node.func)