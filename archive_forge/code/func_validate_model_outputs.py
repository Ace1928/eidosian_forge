import warnings
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union
import numpy as np
from packaging.version import Version, parse
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import (
from .config import OnnxConfig
def validate_model_outputs(config: OnnxConfig, preprocessor: Union['PreTrainedTokenizer', 'FeatureExtractionMixin', 'ProcessorMixin'], reference_model: Union['PreTrainedModel', 'TFPreTrainedModel'], onnx_model: Path, onnx_named_outputs: List[str], atol: float, tokenizer: 'PreTrainedTokenizer'=None):
    from onnxruntime import InferenceSession, SessionOptions
    logger.info('Validating ONNX model...')
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError('You cannot provide both a tokenizer and a preprocessor to validate the model outputs.')
    if tokenizer is not None:
        warnings.warn('The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use `preprocessor` instead.', FutureWarning)
        logger.info('Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.')
        preprocessor = tokenizer
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model_inputs = config.generate_dummy_inputs(preprocessor, batch_size=config.default_fixed_batch + 1, seq_length=config.default_fixed_sequence + 1, framework=TensorType.PYTORCH)
    else:
        reference_model_inputs = config.generate_dummy_inputs(preprocessor, batch_size=config.default_fixed_batch + 1, seq_length=config.default_fixed_sequence + 1, framework=TensorType.TENSORFLOW)
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options, providers=['CPUExecutionProvider'])
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model.to('cpu')
    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}
    for name, value in ref_outputs.items():
        if name == 'past_key_values':
            name = 'present'
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value
    reference_model_inputs_onnxruntime = config.generate_dummy_inputs_onnxruntime(reference_model_inputs)
    onnx_inputs = {}
    for name, value in reference_model_inputs_onnxruntime.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
        else:
            onnx_inputs[name] = value.numpy()
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)
    ref_outputs_set, onnx_outputs_set = (set(ref_outputs_dict.keys()), set(onnx_named_outputs))
    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(f'\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}')
        raise ValueError(f"Outputs doesn't match between reference model and ONNX exported model: {onnx_outputs_set.difference(ref_outputs_set)}")
    else:
        logger.info(f'\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})')
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')
        if not ort_value.shape == ref_value.shape:
            logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(f"Outputs shape doesn't match between reference model and ONNX exported model: Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)")
        else:
            logger.info(f'\t\t-[✓] {ort_value.shape} matches {ref_value.shape}')
        if not np.allclose(ref_value, ort_value, atol=atol):
            bad_indices = np.logical_not(np.isclose(ref_value, ort_value, atol=atol))
            logger.info(f'\t\t-[x] values not close enough (atol: {atol})')
            raise ValueError(f"Outputs values doesn't match between reference model and ONNX exported model: Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))} for {ref_value[bad_indices]} vs {ort_value[bad_indices]}")
        else:
            logger.info(f'\t\t-[✓] all values close (atol: {atol})')