import copy
import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
def visit_comprehension(self, node):
    node.iter = self.visit(node.iter)
    node.target = self.visit(node.target)
    return self.generic_visit(node)