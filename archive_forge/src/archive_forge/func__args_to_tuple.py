import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging
def _args_to_tuple(self, node):
    """Ties together all positional and *arg arguments in a single tuple."""
    builder = _ArgTemplateBuilder()
    for a in node.args:
        if isinstance(a, gast.Starred):
            builder.add_stararg(a.value)
        else:
            builder.add_arg(a)
    builder.finalize()
    return builder.to_ast()