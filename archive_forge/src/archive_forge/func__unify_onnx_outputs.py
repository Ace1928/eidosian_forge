import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Set, Tuple
import numpy as np
import onnx
from onnx import ModelProto, ValueInfoProto, numpy_helper
from ..utils import logging, recurse_getattr
def _unify_onnx_outputs(model1: ModelProto, model2: ModelProto, strict: bool):
    """
    Unifies the outputs of two ONNX model protos. The outputs of model1 will be replaced by outputs of model2.
    According to the rules of "If" op, two subgraphs must have the same number of outputs.
    """
    model1_outputs = {output.name for output in model1.graph.output}
    model2_outputs = {output.name for output in model2.graph.output}
    if model1_outputs != model2_outputs:
        if strict is True:
            raise ValueError(f'The two model protos outputs are expected to have the same number of outputs and output names when strict=True. Found the outputs {model1_outputs - model2_outputs} only in model1, and {model2_outputs - model1_outputs} only in model2.')
        else:
            logger.info(f'The two models proto have different outputs ({len(model1_outputs)} and {len(model2_outputs)} outputs). Constant outputs will be added to unify the two models outputs.')
    if model2_outputs.issubset(model1_outputs) is False:
        raise ValueError('The second ModelProto should not have more outputs than the first.')
    for idx in range(len(model1.graph.output)):
        model_output_1 = model1.graph.output[idx]
        model_output_2 = model2.graph.output[idx] if idx < len(model2.graph.output) else None
        if model_output_2 is None or model_output_1 != model_output_2:
            if model_output_2 is None or not (model_output_1.name == model_output_2.name and model_output_1.type.tensor_type.elem_type == model_output_2.type.tensor_type.elem_type):
                if strict is False and model_output_1.name not in model2_outputs:
                    data_type = model_output_1.type.tensor_type.elem_type
                    dims_output_1 = _infer_output_shape(model_output_1)
                    if not any((isinstance(dim_output, str) for dim_output in dims_output_1)):
                        raise ValueError(f'Expected at least one dynamic input shape for the output {model_output_1.name}, found a static shape: {dims_output_1}')
                    dims_dummy_output = []
                    dummy_axis = None
                    for j, dim in enumerate(dims_output_1):
                        if isinstance(dim, str) and dummy_axis is None:
                            dims_dummy_output.append(0)
                            dummy_axis = j
                        elif isinstance(dim, str) and dummy_axis is not None:
                            dims_dummy_output.append(1)
                        else:
                            dims_dummy_output.append(dim)
                    logger.info(f'Adding a constant output for {model_output_1.name} of shape {dims_dummy_output} in model2.')
                    value = onnx.helper.make_tensor(name='const_tensor', data_type=data_type, dims=dims_dummy_output, vals=[])
                    constant_node = onnx.helper.make_node('Constant', name=f'Constant_{len(model2.graph.node) + 1}', inputs=[], outputs=[f'{model_output_1.name}'], value=value)
                    model2.graph.node.append(constant_node)
                    constant_empty_output = onnx.helper.make_tensor_value_info(model_output_1.name, model_output_1.type.tensor_type.elem_type, _infer_output_shape(model_output_1))
                    model2.graph.output.insert(idx, constant_empty_output)
                elif model_output_2 is not None:
                    raise ValueError(f'Cannot match {model_output_1.name} with {model_output_2.name}. Make sure your model protos have same outputs, have same data types and are in the same order.')
                else:
                    raise ValueError(f'Too few outputs of model2 were found to match with {model_output_1.name}. Please try to pass strict=False, or fill a bug report at https://github.com/huggingface/optimum.')
            else:
                model2.graph.output.remove(model_output_2)
                new_output = onnx.helper.make_tensor_value_info(model_output_1.name, model_output_1.type.tensor_type.elem_type, _infer_output_shape(model_output_1))
                model2.graph.output.insert(idx, new_output)
    if not all((model_output_1 == model_output_2 for model_output_1, model_output_2 in zip(model1.graph.output, model2.graph.output))):
        raise RuntimeError('Failed to unify outputs of given ONNX model protos.')