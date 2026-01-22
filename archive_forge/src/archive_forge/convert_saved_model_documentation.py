from tensorflow.lite.python import util
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
Converts a SavedModel to a frozen graph.

  Args:
    saved_model_dir: SavedModel directory to convert.
    input_arrays: List of input tensors to freeze graph with. Uses input arrays
      from SignatureDef when none are provided.
    input_shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).
      Automatically determined when input shapes is None (e.g., {"foo" : None}).
    output_arrays: List of output tensors to freeze graph with. Uses output
      arrays from SignatureDef when none are provided.
    tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze. All tags in the tag set must be present.
    signature_key: Key identifying SignatureDef containing inputs and outputs.

  Returns:
    frozen_graph_def: Frozen GraphDef.
    in_tensors: List of input tensors for the graph.
    out_tensors: List of output tensors for the graph.
    graph: `Graph` object.

  Raises:
    ValueError:
      SavedModel doesn't contain a MetaGraphDef identified by tag_set.
      signature_key is not in the MetaGraphDef.
      assets/ directory is in the MetaGraphDef.
      input_shapes does not match the length of input_arrays.
      input_arrays or output_arrays are not valid.
  