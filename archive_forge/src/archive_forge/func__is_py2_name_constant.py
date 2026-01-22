import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _is_py2_name_constant(node):
    return isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']