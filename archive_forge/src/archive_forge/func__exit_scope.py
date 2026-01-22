import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def _exit_scope(self):
    exited_scope = self.scope
    exited_scope.finalize()
    self.scope = exited_scope.parent
    return exited_scope