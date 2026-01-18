import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def visit_SetComp(self, node):
    msg = 'SetComp nodes not supported (need to convert to a form that tolerates assignment statements in clause bodies).'
    raise ValueError(msg)