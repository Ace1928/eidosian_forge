import enum
import functools
import pprint
import shutil
import sys
import tempfile
import time
import warnings
from absl import logging
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op  # pylint: disable=unused-import
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metdata_fb
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import convert_graphdef as _convert_graphdef
from tensorflow.lite.python.convert import convert_graphdef_with_arrays as _convert_graphdef_with_arrays
from tensorflow.lite.python.convert import convert_jax_hlo as _convert_jax_hlo
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.lite.python.convert import ConverterError  # pylint: disable=unused-import
from tensorflow.lite.python.convert import deduplicate_readonly_buffers as _deduplicate_readonly_buffers
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import load_delegate  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import OpResolverType  # pylint: disable=unused-import
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import _xla_computation
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_sparsity_modes as _get_sparsity_modes
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import get_tf_type_name as _get_tf_type_name
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import model_input_signature as _model_input_signature
from tensorflow.lite.python.util import modify_model_io_type as _modify_model_io_type
from tensorflow.lite.python.util import populate_conversion_metadata as _populate_conversion_metadata
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.lite.python.util import trace_model_call as _trace_model_call
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugger  # pylint: disable=unused-import
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugOptions  # pylint: disable=unused-import
from tensorflow.python import saved_model as _saved_model
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import save_options as _save_options
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util import keras_deps
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export(v1=['lite.TFLiteConverter'])
class TFLiteConverter(TFLiteFrozenGraphConverter):
    """Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of target
      device, including supported ops set, supported types and a set of user's
      defined TensorFlow operators required in the TensorFlow Lite runtime.
      Refer to `tf.lite.TargetSpec`.
    inference_type: Data type of numeric arrays, excluding the input layer.
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    inference_input_type: Data type of the numeric arrays in the input layer. If
      `inference_input_type` is in {tf.int8, tf.uint8}, then
      `quantized_input_stats` must be provided. (default is the value assigned
      to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the numeric arrays in the output layer.
      (default is the value assigned to `inference_type`, must be in
      {tf.float32, tf.int8, tf.uint8})
    quantized_input_stats: Map of input tensor names to a tuple of floats
      representing the mean and standard deviation of the training data. (e.g.,
      {"foo" : (0., 1.)}). Required if `inference_input_type` is tf.int8 or
      tf.uint8. (default None)
    default_ranges_stats: Tuple of integers (min, max) representing range values
      for all numeric arrays without a specified range. Intended for
      experimenting with quantization via "dummy quantization". (default None)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    output_format: Output file format. (default
      tf.compat.v1.lite.constants.TFLITE, must be in
      {tf.compat.v1.lite.constants.TFLITE,
      tf.compat.v1.lite.constants.GRAPHVIZ_DOT})
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      `output_format=tf.compat.v1.lite.constants.GRAPHVIZ_DOT` in order to keep
      the requirements of the output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the GraphViz .dot
      files after every graph transformation. Requires the `dump_graphviz_dir`
      flag to be specified. (default False)
    conversion_summary_dir: Full path of the directory to store conversion logs.
      (default None)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    target_ops: Deprecated. Please use `target_spec.supported_ops` instead.
    post_training_quantize: Deprecated. Please use `optimizations` instead and
      set it to `{tf.lite.Optimize.DEFAULT}`. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)  Example usage:  ```python # Converting a GraphDef from
      session. converter = tf.compat.v1.lite.TFLiteConverter.from_session( sess,
      in_tensors, out_tensors) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      GraphDef from file. converter =
      tf.compat.v1.lite.TFLiteConverter.from_frozen_graph( graph_def_file,
      input_arrays, output_arrays) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      SavedModel. converter =
      tf.compat.v1.lite.TFLiteConverter.from_saved_model( saved_model_dir)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model)  # Converting a tf.keras model. converter =
      tf.compat.v1.lite.TFLiteConverter.from_keras_model_file( keras_model)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model) ```
  """

    def __init__(self, graph_def, input_tensors, output_tensors, input_arrays_with_shape=None, output_arrays=None, experimental_debug_info_func=None):
        """Constructor for TFLiteConverter.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes (e.g., [("foo" : [1, 16,
        16, 3])]). Use only when graph cannot be loaded into TensorFlow and when
        `input_tensors` and `output_tensors` are None. (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
        super(TFLiteConverter, self).__init__(graph_def, input_tensors, output_tensors, input_arrays_with_shape, output_arrays, experimental_debug_info_func)

    @classmethod
    def from_session(cls, sess, input_tensors, output_tensors):
        """Creates a TFLiteConverter class from a TensorFlow Session.

    Args:
      sess: TensorFlow Session.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).

    Returns:
      TFLiteConverter class.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SESSION)
        graph_def = _freeze_graph(sess, input_tensors, output_tensors)
        return cls(graph_def, input_tensors, output_tensors, experimental_debug_info_func=_build_debug_info_func(sess.graph))

    @classmethod
    def from_frozen_graph(cls, graph_def_file, input_arrays, output_arrays, input_shapes=None):
        """Creates a TFLiteConverter class from a file containing a frozen GraphDef.

    Args:
      graph_def_file: Full filepath of file containing frozen GraphDef.
      input_arrays: List of input tensors to freeze graph with.
      output_arrays: List of output tensors to freeze graph with.
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)

    Returns:
      TFLiteConverter class.

    Raises:
      IOError:
        File not found.
        Unable to parse input file.
      ValueError:
        The graph is not frozen.
        input_arrays or output_arrays contains an invalid tensor name.
        input_shapes is not correctly defined when required
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_GRAPH_DEF)
        with _ops.Graph().as_default():
            with _session.Session() as sess:
                if not gfile.Exists(graph_def_file):
                    raise IOError("File '{0}' does not exist.".format(graph_def_file))
                with gfile.GFile(graph_def_file, 'rb') as f:
                    file_content = f.read()
                try:
                    graph_def = _graph_pb2.GraphDef()
                    graph_def.ParseFromString(file_content)
                except (_text_format.ParseError, DecodeError):
                    try:
                        print("Ignore 'tcmalloc: large alloc' warnings.")
                        if not isinstance(file_content, str):
                            file_content = file_content.decode('utf-8')
                        graph_def = _graph_pb2.GraphDef()
                        _text_format.Merge(file_content, graph_def)
                    except (_text_format.ParseError, DecodeError):
                        raise IOError("Unable to parse input file '{}'.".format(graph_def_file))
                if sys.byteorder == 'big':
                    bst.swap_tensor_content_in_graph_node(graph_def, 'little', 'big')
                load_model_in_session = True
                try:
                    _import_graph_def(graph_def, name='')
                except _NotFoundError:
                    load_model_in_session = False
                if load_model_in_session:
                    if not _is_frozen_graph(sess):
                        raise ValueError('Please freeze the graph using freeze_graph.py.')
                    input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
                    output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
                    _set_tensor_shapes(input_tensors, input_shapes)
                    return cls(sess.graph_def, input_tensors, output_tensors)
                else:
                    if not input_shapes:
                        raise ValueError('input_shapes must be defined for this model.')
                    if set(input_arrays) != set(input_shapes.keys()):
                        raise ValueError('input_shapes must contain a value for each item in input_array.')
                    input_arrays_with_shape = [(name, input_shapes[name]) for name in input_arrays]
                    return cls(graph_def, input_tensors=None, output_tensors=None, input_arrays_with_shape=input_arrays_with_shape, output_arrays=output_arrays)

    @classmethod
    def from_saved_model(cls, saved_model_dir, input_arrays=None, input_shapes=None, output_arrays=None, tag_set=None, signature_key=None):
        """Creates a TFLiteConverter class from a SavedModel.

    Args:
      saved_model_dir: SavedModel directory to convert.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING})
      signature_key: Key identifying SignatureDef containing inputs and outputs.
        (default tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    Returns:
      TFLiteConverter class.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SAVED_MODEL)
        if tag_set is None:
            tag_set = set([_tag_constants.SERVING])
        if signature_key is None:
            signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        saved_model_converter = TFLiteSavedModelConverter(saved_model_dir, tag_set, [signature_key])
        if saved_model_converter.saved_model_dir:
            return saved_model_converter
        result = _freeze_saved_model(saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set, signature_key)
        return cls(graph_def=result[0], input_tensors=result[1], output_tensors=result[2], experimental_debug_info_func=_build_debug_info_func(result[3]))

    @classmethod
    def from_keras_model_file(cls, model_file, input_arrays=None, input_shapes=None, output_arrays=None, custom_objects=None):
        """Creates a TFLiteConverter class from a tf.keras model file.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      custom_objects: Dict mapping names (strings) to custom classes or
        functions to be considered during model deserialization. (default None)

    Returns:
      TFLiteConverter class.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.KERAS_MODEL)
        return TFLiteKerasModelConverter(model_file, input_arrays, input_shapes, output_arrays, custom_objects)

    def convert(self):
        """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
        return super(TFLiteConverter, self).convert()