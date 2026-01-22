import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _do_transform_node(self, node):
    temp_name = self._gensym.new_name()
    temp_assign = templates.replace('temp_name = expr', temp_name=temp_name, expr=node)[0]
    self._add_pending_statement(temp_assign)
    answer = templates.replace('temp_name', temp_name=temp_name)[0]
    return answer