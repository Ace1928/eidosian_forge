import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
def _get_block_composite_vars(self, modified, live_in):
    composite_scope_vars = []
    for s in modified:
        if not s.is_composite():
            continue
        support_set_symbols = tuple((sss for sss in s.support_set if sss.is_symbol()))
        if not all((sss in live_in for sss in support_set_symbols)):
            continue
        composite_scope_vars.append(s)
    return frozenset(composite_scope_vars)