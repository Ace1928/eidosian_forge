import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def _build_argument_parsers(self, config):
    """Build argument parsers for DebugAnalayzer.

    Args:
      config: A `cli_config.CLIConfig` object.

    Returns:
      A dict mapping command handler name to `ArgumentParser` instance.
    """
    self._arg_parsers = {}
    ap = argparse.ArgumentParser(description='List dumped intermediate tensors.', usage=argparse.SUPPRESS)
    ap.add_argument('-f', '--tensor_filter', dest='tensor_filter', type=str, default='', help='List only Tensors passing the filter of the specified name')
    ap.add_argument('-fenn', '--filter_exclude_node_names', dest='filter_exclude_node_names', type=str, default='', help='When applying the tensor filter, exclude node with names matching the regular expression. Applicable only if --tensor_filter or -f is used.')
    ap.add_argument('-n', '--node_name_filter', dest='node_name_filter', type=str, default='', help='filter node name by regex.')
    ap.add_argument('-t', '--op_type_filter', dest='op_type_filter', type=str, default='', help='filter op type by regex.')
    ap.add_argument('-s', '--sort_by', dest='sort_by', type=str, default=SORT_TENSORS_BY_TIMESTAMP, help='the field to sort the data by: (%s | %s | %s | %s)' % (SORT_TENSORS_BY_TIMESTAMP, SORT_TENSORS_BY_DUMP_SIZE, SORT_TENSORS_BY_OP_TYPE, SORT_TENSORS_BY_TENSOR_NAME))
    ap.add_argument('-r', '--reverse', dest='reverse', action='store_true', help='sort the data in reverse (descending) order')
    self._arg_parsers['list_tensors'] = ap
    ap = argparse.ArgumentParser(description='Show information about a node.', usage=argparse.SUPPRESS)
    ap.add_argument('node_name', type=str, help='Name of the node or an associated tensor, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
    ap.add_argument('-a', '--attributes', dest='attributes', action='store_true', help='Also list attributes of the node.')
    ap.add_argument('-d', '--dumps', dest='dumps', action='store_true', help='Also list dumps available from the node.')
    ap.add_argument('-t', '--traceback', dest='traceback', action='store_true', help="Also include the traceback of the node's creation (if available in Python).")
    self._arg_parsers['node_info'] = ap
    ap = argparse.ArgumentParser(description='Show inputs to a node.', usage=argparse.SUPPRESS)
    ap.add_argument('node_name', type=str, help='Name of the node or an output tensor from the node, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
    ap.add_argument('-c', '--control', action='store_true', help='Include control inputs.')
    ap.add_argument('-d', '--depth', dest='depth', type=int, default=config.get('graph_recursion_depth'), help='Maximum depth of recursion used when showing the input tree.')
    ap.add_argument('-r', '--recursive', dest='recursive', action='store_true', help='Show inputs to the node recursively, i.e., the input tree.')
    ap.add_argument('-t', '--op_type', action='store_true', help='Show op types of input nodes.')
    self._arg_parsers['list_inputs'] = ap
    ap = argparse.ArgumentParser(description='Show the nodes that receive the outputs of given node.', usage=argparse.SUPPRESS)
    ap.add_argument('node_name', type=str, help='Name of the node or an output tensor from the node, e.g., hidden1/Wx_plus_b/MatMul, hidden1/Wx_plus_b/MatMul:0')
    ap.add_argument('-c', '--control', action='store_true', help='Include control inputs.')
    ap.add_argument('-d', '--depth', dest='depth', type=int, default=config.get('graph_recursion_depth'), help='Maximum depth of recursion used when showing the output tree.')
    ap.add_argument('-r', '--recursive', dest='recursive', action='store_true', help='Show recipients of the node recursively, i.e., the output tree.')
    ap.add_argument('-t', '--op_type', action='store_true', help='Show op types of recipient nodes.')
    self._arg_parsers['list_outputs'] = ap
    self._arg_parsers['print_tensor'] = command_parser.get_print_tensor_argparser('Print the value of a dumped tensor.')
    ap = argparse.ArgumentParser(description='Print a Python source file with overlaid debug information, including the nodes (ops) or Tensors created at the source lines.', usage=argparse.SUPPRESS)
    ap.add_argument('source_file_path', type=str, help='Path to the source file.')
    ap.add_argument('-t', '--tensors', dest='tensors', action='store_true', help='Label lines with dumped Tensors, instead of ops.')
    ap.add_argument('-m', '--max_elements_per_line', type=int, default=10, help='Maximum number of elements (ops or Tensors) to show per source line.')
    ap.add_argument('-b', '--line_begin', type=int, default=1, help='Print source beginning at line number (1-based.)')
    self._arg_parsers['print_source'] = ap
    ap = argparse.ArgumentParser(description='List source files responsible for constructing nodes and tensors present in the run().', usage=argparse.SUPPRESS)
    ap.add_argument('-p', '--path_filter', type=str, default='', help='Regular expression filter for file path.')
    ap.add_argument('-n', '--node_name_filter', type=str, default='', help='Regular expression filter for node name.')
    self._arg_parsers['list_source'] = ap
    ap = argparse.ArgumentParser(description='Evaluate an arbitrary expression. Can use tensor values\n        from the current debug dump. The debug tensor names should be enclosed\n        in pairs of backticks. Expressions with spaces should be enclosed in\n        a pair of double quotes or a pair of single quotes. By default, numpy\n        is imported as np and can be used in the expressions. E.g.,\n          1) eval np.argmax(`Softmax:0`),\n          2) eval \'np.sum(`Softmax:0`, axis=1)\',\n          3) eval "np.matmul((`output/Identity:0`/`Softmax:0`).T, `Softmax:0`)".\n        ', usage=argparse.SUPPRESS)
    ap.add_argument('expression', type=str, help='Expression to be evaluated.\n        1) in the simplest case, use <node_name>:<output_slot>, e.g.,\n          hidden_0/MatMul:0.\n\n        2) if the default debug op "DebugIdentity" is to be overridden, use\n          <node_name>:<output_slot>:<debug_op>, e.g.,\n          hidden_0/MatMul:0:DebugNumericSummary.\n\n        3) if the tensor of the same name exists on more than one device, use\n          <device_name>:<node_name>:<output_slot>[:<debug_op>], e.g.,\n          /job:worker/replica:0/task:0/gpu:0:hidden_0/MatMul:0\n          /job:worker/replica:0/task:2/cpu:0:hidden_0/MatMul:0:DebugNanCount.\n\n        4) if the tensor is executed multiple times in a given `Session.run`\n        call, specify the execution index with a 0-based integer enclose in a\n        pair of brackets at the end, e.g.,\n          RNN/tanh:0[0]\n          /job:worker/replica:0/task:0/gpu:0:RNN/tanh:0[0].')
    ap.add_argument('-a', '--all', dest='print_all', action='store_true', help='Print the tensor in its entirety, i.e., do not use ellipses (may be slow for large results).')
    ap.add_argument('-w', '--write_path', default='', help='Path of the numpy file to write the evaluation result to, using numpy.save()')
    self._arg_parsers['eval'] = ap