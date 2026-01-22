import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def _function_scope_options(self, fn_scope):
    """Returns the options with which to create function scopes."""
    if fn_scope.level == 2:
        return self.ctx.user.options
    return self.ctx.user.options.call_options()