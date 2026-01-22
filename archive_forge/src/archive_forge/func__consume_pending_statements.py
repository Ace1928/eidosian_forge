import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _consume_pending_statements(self):
    ans = self._pending_statements
    self._pending_statements = []
    return ans