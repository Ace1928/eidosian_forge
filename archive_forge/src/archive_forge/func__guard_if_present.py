from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _guard_if_present(self, block, var_name):
    """Prevents the block from executing if var_name is set."""
    if not block:
        return block
    template = '\n        if not var_name:\n          block\n      '
    node = templates.replace(template, var_name=var_name, block=block)
    return node