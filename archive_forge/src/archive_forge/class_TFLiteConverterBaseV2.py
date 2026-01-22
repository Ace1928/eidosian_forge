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
class TFLiteConverterBaseV2(TFLiteConverterBase):
    """Converter subclass to share functionality between V2 converters."""

    def __init__(self):
        """Constructor for TFLiteConverter."""
        super(TFLiteConverterBaseV2, self).__init__()
        self.inference_input_type = _dtypes.float32
        self.inference_output_type = _dtypes.float32
        self._metadata.environment.apiVersion = 2

    def _validate_inference_input_output_types(self, quant_mode):
        """Validate inference_input_type and inference_output_type flags."""
        default_types = [_dtypes.float32]
        if quant_mode.is_integer_quantization():
            if quant_mode.is_post_training_int16x8_quantization():
                all_types = default_types + [_dtypes.int16]
            else:
                all_types = default_types + [_dtypes.int8, _dtypes.uint8]
            if self.inference_input_type not in all_types or self.inference_output_type not in all_types:
                all_types_names = ['tf.' + t.name for t in all_types]
                raise ValueError('The inference_input_type and inference_output_type must be in {}.'.format(all_types_names))
        elif self.inference_input_type not in default_types or self.inference_output_type not in default_types:
            raise ValueError('The inference_input_type and inference_output_type must be tf.float32.')

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.LOAD_SAVED_MODEL)
    def _load_saved_model(self, saved_model_dir, saved_model_tags):
        """Load graph_def from saved model with the default serving signature key.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze.

    Returns:
      graph_def: The loaded GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
    """
        graph = _ops.Graph()
        saved_model = _loader_impl.SavedModelLoader(saved_model_dir)
        saved_model.load_graph(graph, tags=saved_model_tags)
        meta_graph = saved_model.get_meta_graph_def_from_tags(saved_model_tags)
        graph_def = meta_graph.graph_def
        signature_def = meta_graph.signature_def[_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensors = [graph.get_tensor_by_name(signature_def.inputs[key].name) for key in signature_def.inputs]
        output_tensors = [graph.get_tensor_by_name(signature_def.outputs[key].name) for key in signature_def.outputs]
        return (graph_def, input_tensors, output_tensors)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
    def _validate_inputs(self, graph_def, input_tensors):
        """Validate the input parameters.

    Args:
      graph_def: The TensorFlow GraphDef.
      input_tensors: List of input tensors.

    Raise:
      ValueError: Input shape is not specified. Invalid quantization parameters.
    """
        self._save_conversion_params_metric(graph_def)
        self._quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        self._validate_inference_input_output_types(self._quant_mode)
        if not self._is_unknown_shapes_allowed():
            for tensor in input_tensors:
                shape_list = tensor.shape.as_list()
                if None in shape_list[1:]:
                    raise ValueError("None is only supported in the 1st dimension. Tensor '{0}' has invalid shape '{1}'.".format(_get_tensor_name(tensor), shape_list))
                elif shape_list and shape_list[0] is None:
                    shape = tensor.shape.as_list()
                    shape[0] = 1
                    tensor.set_shape(shape)
        if self._trackable_obj is None or not hasattr(self._trackable_obj, 'graph_debug_info'):
            self._debug_info = _get_debug_info(_build_debug_info_func(self._funcs[0].graph), graph_def)
        else:
            self._debug_info = _get_debug_info(_convert_debug_info_func(self._trackable_obj.graph_debug_info), graph_def)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
    def _optimize_tf_model(self, graph_def, input_tensors, output_tensors, frozen_func):
        """Run a Grappler pass to optimize the TensorFlow graph.

    Args:
      graph_def: Frozen GraphDef to be optimized.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      frozen_func: TensorFlow Graph.

    Returns:
      The optimized TensorFlow graph.
    """
        grappler_config = self._grappler_config()
        if grappler_config.graph_options.rewrite_options.optimizers:
            graph_def = _run_graph_optimizations(graph_def, input_tensors, output_tensors, config=grappler_config, graph=frozen_func.graph)
        return graph_def

    def _convert_from_saved_model(self, graph_def):
        """Helper method that converts saved model.

    Args:
      graph_def: GraphDef object for the model, used only for stats.

    Returns:
      The converted TFLite model.
    """
        self._save_conversion_params_metric(graph_def)
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        self._validate_inference_input_output_types(quant_mode)
        converter_kwargs = {'enable_tflite_resource_variables': self.experimental_enable_resource_variables}
        converter_kwargs.update(self._get_base_converter_args())
        converter_kwargs.update(quant_mode.converter_flags())
        result = _convert_saved_model(**converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)

    def convert(self, graph_def, input_tensors, output_tensors):
        """Converts a TensorFlow GraphDef based on instance variables.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
        self._validate_inputs(graph_def, input_tensors)
        converter_kwargs = self._get_base_converter_args()
        converter_kwargs.update(self._quant_mode.converter_flags())
        if not self.experimental_new_converter:
            logging.warning('Please consider switching to the new converter by setting experimental_new_converter=True. The old converter is deprecated.')
        else:
            logging.info('Using new converter: If you encounter a problem please file a bug. You can opt-out by setting experimental_new_converter=False')
        result = _convert_graphdef(input_data=graph_def, input_tensors=input_tensors, output_tensors=output_tensors, **converter_kwargs)
        return self._optimize_tflite_model(result, self._quant_mode, quant_io=self.experimental_new_quantizer)