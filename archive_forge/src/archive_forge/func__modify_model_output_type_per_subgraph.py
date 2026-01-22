import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def _modify_model_output_type_per_subgraph(model, subgraph_index, signature_index, inference_output_type):
    """Modify model output type per subgraph."""
    subgraph = model.subgraphs[subgraph_index]
    tensors = subgraph.tensors
    operators = subgraph.operators
    dequant_opcode_idxs = get_dequantize_opcode_idx(model)
    if operators and (not dequant_opcode_idxs):
        for output in subgraph.outputs:
            output_type = _convert_tflite_enum_type_to_tf_type(tensors[output].type)
            if output_type == dtypes.float32:
                raise ValueError('Model output is not dequantized.')
        return
    output_dequant_ops = []
    for op in operators:
        if op.opcodeIndex in dequant_opcode_idxs and op.outputs[0] in subgraph.outputs:
            quant_tensor, float_tensor = (tensors[op.inputs[0]], tensors[op.outputs[0]])
            float_type = _convert_tflite_enum_type_to_tf_type(float_tensor.type)
            if float_type != dtypes.float32:
                if float_type == inference_output_type:
                    continue
                else:
                    raise ValueError("Initial model output type must be tf.float32. Expected type for tensor with name '{}' is tf.float32, instead type is {}".format(float_tensor.name, get_tf_type_name(float_type)))
            quant_type = _convert_tflite_enum_type_to_tf_type(quant_tensor.type)
            if quant_type not in _MAP_QUANT_TO_IO_TYPES:
                raise ValueError("Initial model output is not dequantized. Expected type for tensor with name '{}' should be in {}, instead type is {}".format(quant_tensor.name, tuple((get_tf_type_name(t) for t in _MAP_QUANT_TO_IO_TYPES.keys())), get_tf_type_name(quant_type)))
            else:
                inference_io_types = _MAP_QUANT_TO_IO_TYPES[quant_type]
                if inference_output_type not in inference_io_types:
                    raise ValueError('Unsupported `inference_output_type` value. Expected to be in {}, instead got {}.'.format(tuple((get_tf_type_name(t) for t in inference_io_types)), get_tf_type_name(inference_output_type)))
            output_dequant_ops.append(op)
    if len(subgraph.outputs) != len(output_dequant_ops):
        logging.warning('For model outputs containing unsupported operations which cannot be quantized, the `inference_output_type` attribute will default to the original type.')
    if inference_output_type == dtypes.uint8:
        quant_opcode_idx = -1
        for idx, opcode in enumerate(model.operatorCodes):
            builtin_code = schema_util.get_builtin_code_from_operator_code(opcode)
            if builtin_code == schema_fb.BuiltinOperator.QUANTIZE:
                quant_opcode_idx = idx
                break
        if quant_opcode_idx == -1:
            quant_op = schema_fb.OperatorCodeT()
            quant_op.builtinCode = schema_fb.BuiltinOperator.QUANTIZE
            quant_op.deprecatedBuiltinCode = schema_fb.BuiltinOperator.QUANTIZE
            model.operatorCodes.append(quant_op)
            quant_opcode_idx = len(model.operatorCodes) - 1
        for op in output_dequant_ops:
            op.opcodeIndex = quant_opcode_idx
            int8_quantization = tensors[op.inputs[0]].quantization
            uint8_quantization = schema_fb.QuantizationParametersT()
            uint8_quantization.scale = [int8_quantization.scale[0]]
            uint8_quantization.zeroPoint = [int8_quantization.zeroPoint[0] + 128]
            tensors[op.outputs[0]].quantization = uint8_quantization
            tensors[op.outputs[0]].type = schema_fb.TensorType.UINT8
    elif inference_output_type in _MAP_QUANT_TO_IO_TYPES:
        remove_tensors_idxs = set()
        for op in output_dequant_ops:
            subgraph.outputs[subgraph.outputs == op.outputs[0]] = op.inputs[0]
            if signature_index >= 0:
                signature_def = model.signatureDefs[signature_index]
                for i in range(len(signature_def.outputs)):
                    if signature_def.outputs[i].tensorIndex == op.outputs[0]:
                        signature_def.outputs[i].tensorIndex = op.inputs[0]
            remove_tensors_idxs.add(op.outputs[0])
            operators.remove(op)
        _remove_tensors_from_model(model, remove_tensors_idxs)
    else:
        raise ValueError('Unsupported `inference_output_type` value {}.'.format(get_tf_type_name(inference_output_type)))