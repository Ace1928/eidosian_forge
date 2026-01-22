import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import numpy as np
from packaging import version
from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size
class OnnxConfigWithPast(OnnxConfig, ABC):

    def __init__(self, config: 'PretrainedConfig', task: str='default', patching_specs: List[PatchingSpec]=None, use_past: bool=False):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: 'PretrainedConfig', task: str='default') -> 'OnnxConfigWithPast':
        """
        Instantiate a OnnxConfig with `use_past` attribute set to True

        Args:
            config: The underlying model's config to use when exporting to ONNX

        Returns:
            OnnxConfig with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction='outputs')
        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, 'use_cache'):
            return {'use_cache': self.use_past}
        return None

    @property
    def num_layers(self) -> int:
        """
        The number of layers attribute retrieved from the model config. Override this for model configs where the
        number of layers attribute is not called `num_layers`.
        """
        if not hasattr(self._config, 'num_layers'):
            raise AttributeError('could not find the number of layers attribute in the model configuration, override the num_layers property of the model OnnxConfig to solve this')
        return self._config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        The number of attention heads attribute retrieved from the model config. Override this for model configs where
        the number of attention heads attribute is not called `num_attention_heads`.
        """
        if not hasattr(self._config, 'num_attention_heads'):
            raise AttributeError('could not find the number of attention heads attribute in the model configuration, override the num_attention_heads property of the model OnnxConfig to solve this')
        return self._config.num_attention_heads

    def generate_dummy_inputs(self, tokenizer: 'PreTrainedTokenizerBase', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
        common_inputs = super().generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        if self.use_past:
            if not is_torch_available():
                raise ValueError('Cannot generate dummy past_keys inputs without PyTorch installed.')
            else:
                import torch
            batch, seqlen = common_inputs['input_ids'].shape
            past_key_values_length = seqlen + 2
            shape = (batch, self.num_attention_heads, past_key_values_length, self._config.hidden_size // self.num_attention_heads)
            if 'attention_mask' in common_inputs:
                mask_dtype = common_inputs['attention_mask'].dtype
                common_inputs['attention_mask'] = torch.cat([common_inputs['attention_mask'], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1)
            common_inputs['past_key_values'] = []
            for _ in range(self.num_layers):
                common_inputs['past_key_values'].append((torch.zeros(shape), torch.zeros(shape)))
        return common_inputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str, inverted_values_shape: bool=False):
        """
        Fill the input_or_outputs mapping with past_key_values dynamic axes considering.

        Args:
            inputs_or_outputs: The mapping to fill.
            direction: either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
                output mapping, this is important for axes naming.
            inverted_values_shape:
                If `True`, store values on dynamic axis 1, else on axis 2.

        """
        if direction not in ['inputs', 'outputs']:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
        name = 'past_key_values' if direction == 'inputs' else 'present'
        for i in range(self.num_layers):
            inputs_or_outputs[f'{name}.{i}.key'] = {0: 'batch', 2: 'past_sequence + sequence'}
            if inverted_values_shape:
                inputs_or_outputs[f'{name}.{i}.value'] = {0: 'batch', 1: 'past_sequence + sequence'}
            else:
                inputs_or_outputs[f'{name}.{i}.value'] = {0: 'batch', 2: 'past_sequence + sequence'}

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f'{name}.{idx}.key'] = t[0]
        flattened_output[f'{name}.{idx}.value'] = t[1]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        flattened_output = {}
        if name in ['present', 'past_key_values']:
            for idx, t in enumerate(field):
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)
        return flattened_output