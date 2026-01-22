import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
def _as_lambda(self, expr):
    return templates.replace_as_expression('lambda: expr', expr=expr)