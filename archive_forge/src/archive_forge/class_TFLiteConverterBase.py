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
class TFLiteConverterBase:
    """Converter superclass to share functionality between V1 and V2 converters."""
    _original_model_type = conversion_metdata_fb.ModelType.NONE

    def __init__(self):
        self.optimizations = set()
        self.representative_dataset = None
        self.target_spec = TargetSpec()
        self.allow_custom_ops = False
        self.experimental_new_converter = True
        self.experimental_new_quantizer = True
        self.experimental_enable_resource_variables = True
        self._experimental_calibrate_only = False
        self._experimental_sparsify_model = False
        self._experimental_disable_per_channel = False
        self._debug_info = None
        self.saved_model_dir = None
        self._saved_model_tags = None
        self._saved_model_version = 0
        self._saved_model_exported_names = []
        self._tflite_metrics = metrics.TFLiteConverterMetrics()
        self._collected_converter_params = {}
        self.unfold_batchmatmul = False
        self.legalize_custom_tensor_list_ops = False
        self._experimental_lower_tensor_list_ops = True
        self._experimental_default_to_single_batch_in_tensor_list_ops = False
        self._experimental_unfold_large_splat_constant = False
        self._experimental_tf_quantization_mode = None
        self._experimental_full_integer_quantization_bias_type = None
        self._experimental_quantization_options = None
        self.exclude_conversion_metadata = False
        self._metadata = conversion_metdata_fb.ConversionMetadataT()
        self._metadata.environment = conversion_metdata_fb.EnvironmentT()
        self._metadata.options = conversion_metdata_fb.ConversionOptionsT()
        self._metadata.environment.tensorflowVersion = versions.__version__
        self._metadata.environment.modelType = self._get_original_model_type()
        self._experimental_enable_dynamic_update_slice = False
        self._experimental_preserve_assert_op = False
        self._experimental_guarantee_all_funcs_one_use = False
        self.experimental_new_dynamic_range_quantizer = True
        self._experimental_low_bit_qat = False
        self._experimental_allow_all_select_tf_ops = False
        self._experimental_variable_quantization = False
        self._experimental_disable_fuse_mul_and_fc = False
        self._experimental_use_buffer_offset = False
        self.mlir_dump_dir = None
        self.mlir_dump_pass_regex = None
        self.mlir_dump_func_regex = None
        self.mlir_enable_timing = None
        self.mlir_print_ir_before = None
        self.mlir_print_ir_after = None
        self.mlir_print_ir_module_scope = None
        self.mlir_elide_elementsattrs_if_larger = None

    def _grappler_config(self, optimizers=None):
        """Creates a tf.compat.v1.ConfigProto for configuring Grappler.

    Args:
      optimizers: List of strings that represents the list of optimizers.

    Returns:
      tf.ConfigProto.
    """
        if not optimizers:
            optimizers = []
        if not self.experimental_new_converter:
            optimizers.append('constfold')
        is_only_flex_enabled = set([OpsSet.SELECT_TF_OPS]) == set(self.target_spec.supported_ops)
        if is_only_flex_enabled:
            optimizers.append('layout')
        return _get_grappler_config(optimizers)

    def _quantize(self, result, input_type, output_type, activations_type, bias_type, allow_float, enable_variable_quantization):
        """Quantize the model."""
        custom_op_registerers_by_name = [x for x in self.target_spec._experimental_custom_op_registerers if isinstance(x, str)]
        custom_op_registerers_by_func = [x for x in self.target_spec._experimental_custom_op_registerers if not isinstance(x, str)]
        if not isinstance(self.representative_dataset, RepresentativeDataset):
            self.representative_dataset = RepresentativeDataset(self.representative_dataset)
        result = _calibrator.add_intermediate_tensors(result)
        calibrate_quantize = _calibrator.Calibrator(result, custom_op_registerers_by_name, custom_op_registerers_by_func)
        if self._experimental_calibrate_only or self.experimental_new_quantizer:
            calibrated = calibrate_quantize.calibrate(self.representative_dataset.input_gen)
        if self._experimental_calibrate_only:
            return calibrated
        elif self.experimental_new_quantizer and activations_type != _dtypes.int16:
            return _mlir_quantize(calibrated, self._experimental_disable_per_channel, input_data_type=input_type, output_data_type=output_type, enable_variable_quantization=enable_variable_quantization)
        else:
            return calibrate_quantize.calibrate_and_quantize(self.representative_dataset.input_gen, input_type, output_type, allow_float, activations_type, bias_type, disable_per_channel=self._experimental_disable_per_channel)

    def _is_unknown_shapes_allowed(self):
        return self.experimental_new_converter

    def _get_base_converter_args(self):
        """Returns the base converter args.

    Returns:
      {key str: val}
    """
        args = {'input_format': constants.TENSORFLOW_GRAPHDEF, 'allow_custom_ops': self.allow_custom_ops, 'debug_info': self._debug_info, 'target_ops': self.target_spec.supported_ops, 'enable_mlir_converter': self.experimental_new_converter, 'select_user_tf_ops': self.target_spec.experimental_select_user_tf_ops, 'supported_backends': self.target_spec.experimental_supported_backends, 'unfold_batchmatmul': self.unfold_batchmatmul, 'legalize_custom_tensor_list_ops': self.legalize_custom_tensor_list_ops, 'lower_tensor_list_ops': self._experimental_lower_tensor_list_ops, 'unfold_large_splat_constant': self._experimental_unfold_large_splat_constant, 'default_to_single_batch_in_tensor_list_ops': self._experimental_default_to_single_batch_in_tensor_list_ops, 'tf_quantization_mode': self._experimental_tf_quantization_mode, 'experimental_enable_resource_variables': self.experimental_enable_resource_variables, 'enable_dynamic_update_slice': self._experimental_enable_dynamic_update_slice, 'preserve_assert_op': self._experimental_preserve_assert_op, 'guarantee_all_funcs_one_use': self._experimental_guarantee_all_funcs_one_use, 'allow_all_select_tf_ops': self._experimental_allow_all_select_tf_ops, 'disable_fuse_mul_and_fc': self._experimental_disable_fuse_mul_and_fc, 'quantization_options': self._experimental_quantization_options, 'mlir_dump_dir': self.mlir_dump_dir, 'mlir_dump_pass_regex': self.mlir_dump_pass_regex, 'mlir_dump_func_regex': self.mlir_dump_func_regex, 'mlir_enable_timing': self.mlir_enable_timing, 'mlir_print_ir_before': self.mlir_print_ir_before, 'mlir_print_ir_after': self.mlir_print_ir_after, 'mlir_print_ir_module_scope': self.mlir_print_ir_module_scope, 'mlir_elide_elementsattrs_if_larger': self.mlir_elide_elementsattrs_if_larger, 'use_buffer_offset': self._experimental_use_buffer_offset}
        if self.saved_model_dir:
            args.update({'saved_model_dir': self.saved_model_dir, 'saved_model_version': self._saved_model_version, 'saved_model_tags': self._saved_model_tags, 'saved_model_exported_names': self._saved_model_exported_names})
        if self._experimental_quantization_options:
            logging.warning('Configs from custom methods in experimental_quantization_options may not produce a valid tflite model. Note that currently this option only supports StableHLO path. Setting this option in TFLite path will be a no-op.')
        return args

    def _contains_function_with_implements_attr(self, saved_model_proto):
        meta_graph = saved_model_proto.meta_graphs[0]
        for function in meta_graph.graph_def.library.function:
            if function.attr.get('_implements', None) or function.attr.get('api_implements', None):
                return True
        return False

    def _parse_saved_model_args(self, always_enable_saved_model_import=False):
        """Parses SavedModel arguments from the given Keras/RNN SavedModel.

    Args:
      always_enable_saved_model_import: Bool. When the value is true, it enables
        MLIR saved model import path regardless of checking the conditions.
    """
        if not self.experimental_new_converter:
            self.saved_model_dir = None
            return
        if self.saved_model_dir:
            try:
                saved_model_proto, _ = _parse_saved_model_with_debug_info(self.saved_model_dir)
            except OSError:
                self.saved_model_dir = None
                return
            if not always_enable_saved_model_import and (not self._contains_function_with_implements_attr(saved_model_proto)):
                self.saved_model_dir = None
                return
            if not self._saved_model_exported_names:
                self._saved_model_exported_names = []
            self._saved_model_version = saved_model_proto.saved_model_schema_version
            if self._saved_model_version == 0:
                self.saved_model_dir = None
                logging.warning('SavedModel schema version is zero.')
                return
            if self._saved_model_version not in [1, 2]:
                raise ValueError('SavedModel file format({0}) is not supported'.format(self._saved_model_version))

    def _sparsify_model(self):
        return Optimize.EXPERIMENTAL_SPARSITY in self.optimizations

    def _increase_conversion_attempt_metric(self):
        self._tflite_metrics.increase_counter_converter_attempt()

    def _increase_conversion_success_metric(self):
        self._tflite_metrics.increase_counter_converter_success()

    @classmethod
    def _set_original_model_type(cls, model_type):
        """Stores the original model type."""
        if model_type == conversion_metdata_fb.ModelType.NONE:
            raise ValueError('The original model type should be specified.')
        cls._original_model_type = model_type

    def _get_original_model_type(self):
        """One-time getter to return original model type and set it to NONE."""
        model_type = TFLiteConverterBase._original_model_type
        TFLiteConverterBase._original_model_type = conversion_metdata_fb.ModelType.NONE
        return model_type

    def _save_conversion_params_metric(self, graph_def=None, inference_type=None, inference_input_type=None):
        """Set conversion parameter metrics."""
        converter_kwargs = self._collected_converter_params
        converter_kwargs.update(self._get_base_converter_args())
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        converter_kwargs.update({'tf_version': self._metadata.environment.tensorflowVersion, 'api_version': self._metadata.environment.apiVersion, 'original_model_format': self._metadata.environment.modelType, 'optimization_default': quant_mode.is_any_optimization_enabled(), 'optimization_post_training_dynamic_range': quant_mode.is_post_training_dynamic_range_quantization(), 'optimization_post_training_float16': quant_mode.is_post_training_float16_quantization(), 'optimization_post_training_integer_quantize': quant_mode.is_post_training_integer_quantization(), 'optimization_qat': quant_mode.is_quantization_aware_training(), 'optimization_low_bit_qat': quant_mode.is_low_bit_quantize_aware_training(), 'optimization_sparsify': self._sparsify_model(), 'activations_type': quant_mode.activations_type()})
        converter_kwargs.update(quant_mode.converter_flags(inference_type, inference_input_type))
        if self.target_spec._experimental_supported_accumulation_type:
            converter_kwargs.update({'accumulation_type': self.target_spec._experimental_supported_accumulation_type})

        def format_element(elem):
            if isinstance(elem, enum.Enum):
                return str(elem.value)
            return pprint.pformat(elem)

        def format_param(param):
            if isinstance(param, (list, tuple, set)):
                if not param:
                    return 'None'
                string_list = [format_element(x) for x in param]
                return ','.join(sorted(string_list))
            return format_element(param)
        for key, value in converter_kwargs.items():
            self._tflite_metrics.set_converter_param(key, format_param(value))
        self._tflite_metrics.set_export_required()
        self._metadata.options.allowCustomOps = self.allow_custom_ops
        self._metadata.options.enableSelectTfOps = OpsSet.SELECT_TF_OPS in self.target_spec.supported_ops
        self._metadata.options.forceSelectTfOps = set([OpsSet.SELECT_TF_OPS]) == set(self.target_spec.supported_ops)
        self._metadata.options.modelOptimizationModes = []
        if quant_mode.is_post_training_float16_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_FLOAT16)
        if quant_mode.is_post_training_dynamic_range_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE)
        if quant_mode.is_post_training_int8_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER)
        if quant_mode.is_post_training_int16x8_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_INT16)
        if quant_mode.is_quantization_aware_training():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.QUANTIZATION_AWARE_TRAINING)

    def _set_conversion_latency_metric(self, value):
        self._tflite_metrics.set_converter_latency(value)

    @convert_phase(Component.OPTIMIZE_TFLITE_MODEL)
    def _optimize_tflite_model(self, model, quant_mode, quant_io=True):
        """Apply optimizations on a TFLite model."""
        if quant_mode.is_integer_quantization():
            in_type, out_type = (self.inference_input_type, self.inference_output_type)
            if quant_mode.is_post_training_integer_quantization():
                q_in_type = in_type if in_type and quant_io else _dtypes.float32
                q_out_type = out_type if out_type and quant_io else _dtypes.float32
                q_activations_type = quant_mode.activations_type()
                q_bias_type = quant_mode.bias_type()
                q_allow_float = quant_mode.is_allow_float()
                q_variable_quantization = quant_mode.enable_mlir_variable_quantization
                model = self._quantize(model, q_in_type, q_out_type, q_activations_type, q_bias_type, q_allow_float, q_variable_quantization)
            m_in_type = in_type if in_type else _dtypes.float32
            m_out_type = out_type if out_type else _dtypes.float32
            if not (quant_mode.is_post_training_integer_quantization() and self.experimental_new_quantizer and quant_io and (m_in_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32]) and (m_out_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32])):
                model = _modify_model_io_type(model, m_in_type, m_out_type)
        if self._sparsify_model():
            model = _mlir_sparsify(model)
        if not self._experimental_use_buffer_offset:
            try:
                model_object = flatbuffer_utils.convert_bytearray_to_object(model)
                if _check_model_use_buffer_offset(model_object):
                    return model
                model = _deduplicate_readonly_buffers(model)
            except Exception:
                logging.warning('Buffer deduplication procedure will be skipped when flatbuffer library is not properly loaded')
        return model

    def _convert_and_export_metrics(self, convert_func, *args, **kwargs):
        """Wraps around convert function to export metrics.

    Args:
      convert_func: The convert function to wrap.
      *args: Positional arguments of the convert function.
      **kwargs: The keyword arguments of the convert function.

    Returns:
      The decorator to wrap the convert function.
    """
        self._increase_conversion_attempt_metric()
        self._save_conversion_params_metric()
        start_time = time.process_time()
        result = convert_func(self, *args, **kwargs)
        elapsed_time_ms = (time.process_time() - start_time) * 1000
        if result:
            self._increase_conversion_success_metric()
        self._set_conversion_latency_metric(round(elapsed_time_ms))
        self._tflite_metrics.export_metrics()
        if self.exclude_conversion_metadata or self._experimental_use_buffer_offset:
            return result
        model_object = flatbuffer_utils.convert_bytearray_to_object(result)
        if _check_model_use_buffer_offset(model_object):
            return result
        sparsity_modes = _get_sparsity_modes(model_object)
        self._metadata.options.modelOptimizationModes.extend(sparsity_modes)
        model_object = _populate_conversion_metadata(model_object, self._metadata)
        return flatbuffer_utils.convert_object_to_bytearray(model_object)