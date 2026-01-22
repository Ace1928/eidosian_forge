import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _generate_pop_operation(self, original_call_node, pop_var_name):
    assert isinstance(original_call_node.func, gast.Attribute)
    if original_call_node.args:
        pop_element = original_call_node.args[0]
    else:
        pop_element = parser.parse_expression('None')
    dtype = self.get_definition_directive(original_call_node.func.value, directives.set_element_type, 'dtype', default=templates.replace_as_expression('None'))
    shape = self.get_definition_directive(original_call_node.func.value, directives.set_element_type, 'shape', default=templates.replace_as_expression('None'))
    template = '\n      target, pop_var_name = ag__.list_pop(\n          target, element,\n          opts=ag__.ListPopOpts(element_dtype=dtype, element_shape=shape))\n    '
    return templates.replace(template, target=original_call_node.func.value, pop_var_name=pop_var_name, element=pop_element, dtype=dtype, shape=shape)