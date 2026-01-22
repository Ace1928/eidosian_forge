import argparse
import os
import sys
import warnings
from absl import app
import tensorflow as tf  # pylint: disable=unused-import
from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.lite.toco.logging import gen_html
from tensorflow.python import tf2
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.util import keras_deps
def _convert_tf1_model(flags):
    """Calls function to convert the TensorFlow 1.X model into a TFLite model.

  Args:
    flags: argparse.Namespace object.

  Raises:
    ValueError: Invalid flags.
  """
    if flags.custom_opdefs:
        register_custom_opdefs(_parse_array(flags.custom_opdefs))
    converter = _get_tflite_converter(flags)
    if flags.inference_type:
        converter.inference_type = _parse_inference_type(flags.inference_type, 'inference_type')
    if flags.inference_input_type:
        converter.inference_input_type = _parse_inference_type(flags.inference_input_type, 'inference_input_type')
    if flags.output_format:
        converter.output_format = _toco_flags_pb2.FileFormat.Value(flags.output_format)
    if flags.mean_values and flags.std_dev_values:
        input_arrays = converter.get_input_arrays()
        std_dev_values = _parse_array(flags.std_dev_values, type_fn=float)
        if converter.inference_type == dtypes.float32:
            mean_values = _parse_array(flags.mean_values, type_fn=float)
        else:
            mean_values = _parse_array(flags.mean_values, type_fn=int)
        quant_stats = list(zip(mean_values, std_dev_values))
        if not flags.input_arrays and len(input_arrays) > 1 or len(input_arrays) != len(quant_stats):
            raise ValueError("Mismatching --input_arrays, --std_dev_values, and --mean_values. The flags must have the same number of items. The current input arrays are '{0}'. --input_arrays must be present when specifying --std_dev_values and --mean_values with multiple input tensors in order to map between names and values.".format(','.join(input_arrays)))
        converter.quantized_input_stats = dict(list(zip(input_arrays, quant_stats)))
    if flags.default_ranges_min is not None and flags.default_ranges_max is not None:
        converter.default_ranges_stats = (flags.default_ranges_min, flags.default_ranges_max)
    if flags.drop_control_dependency:
        converter.drop_control_dependency = flags.drop_control_dependency
    if flags.reorder_across_fake_quant:
        converter.reorder_across_fake_quant = flags.reorder_across_fake_quant
    if flags.change_concat_input_ranges:
        converter.change_concat_input_ranges = flags.change_concat_input_ranges == 'TRUE'
    if flags.allow_custom_ops:
        converter.allow_custom_ops = flags.allow_custom_ops
    if flags.target_ops:
        ops_set_options = lite.OpsSet.get_options()
        converter.target_spec.supported_ops = set()
        for option in flags.target_ops.split(','):
            if option not in ops_set_options:
                raise ValueError('Invalid value for --target_ops. Options: {0}'.format(','.join(ops_set_options)))
            converter.target_spec.supported_ops.add(lite.OpsSet(option))
    if flags.experimental_select_user_tf_ops:
        if lite.OpsSet.SELECT_TF_OPS not in converter.target_spec.supported_ops:
            raise ValueError('--experimental_select_user_tf_ops can only be set if --target_ops contains SELECT_TF_OPS.')
        user_op_set = set()
        for op_name in flags.experimental_select_user_tf_ops.split(','):
            user_op_set.add(op_name)
        converter.target_spec.experimental_select_user_tf_ops = list(user_op_set)
    if flags.post_training_quantize:
        converter.optimizations = [lite.Optimize.DEFAULT]
        if converter.inference_type != dtypes.float32:
            print('--post_training_quantize quantizes a graph of inference_type FLOAT. Overriding inference_type to FLOAT.')
            converter.inference_type = dtypes.float32
    if flags.quantize_to_float16:
        converter.target_spec.supported_types = [dtypes.float16]
        if not flags.post_training_quantize:
            print('--quantize_to_float16 will only take effect with the --post_training_quantize flag enabled.')
    if flags.dump_graphviz_dir:
        converter.dump_graphviz_dir = flags.dump_graphviz_dir
    if flags.dump_graphviz_video:
        converter.dump_graphviz_vode = flags.dump_graphviz_video
    if flags.conversion_summary_dir:
        converter.conversion_summary_dir = flags.conversion_summary_dir
    converter.experimental_new_converter = flags.experimental_new_converter
    if flags.experimental_new_quantizer is not None:
        converter.experimental_new_quantizer = flags.experimental_new_quantizer
    output_data = converter.convert()
    with gfile.GFile(flags.output_file, 'wb') as f:
        f.write(output_data)