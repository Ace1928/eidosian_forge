from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
class WhileBodyFuncGraph(ControlFlowFuncGraph):
    """FuncGraph for the body of tf.while_loop().

  This is used to distinguish while bodies from other functions.
  """