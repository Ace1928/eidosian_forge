import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import templates
def _process_single_assignment(self, target, value):
    if not isinstance(target, gast.Subscript):
        return None
    s = target.slice
    if isinstance(s, (gast.Tuple, gast.Slice)):
        return None
    template = '\n      target = ag__.set_item(target, key, item)\n    '
    return templates.replace(template, target=target.value, key=target.slice, item=value)