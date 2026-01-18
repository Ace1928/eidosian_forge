import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging
def to_ast(self):
    assert self._finalized
    if self._argspec:
        result = self._argspec[0]
        for i in range(1, len(self._argspec)):
            result = gast.BinOp(result, gast.Add(), self._argspec[i])
        return result
    return gast.Tuple([], gast.Load())