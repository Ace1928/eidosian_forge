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
class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        for name, axes_names in common_outputs.items():
            sequence_name = 'encoder_sequence' if 'encoder' in name else 'decoder_sequence'
            for axis_idx, name in axes_names.items():
                if 'sequence' in name:
                    axes_names[axis_idx] = sequence_name
                else:
                    axes_names[axis_idx] = name
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction='outputs')
        return common_outputs

    @property
    def num_layers(self) -> Tuple[int]:
        try:
            num_layers = super().num_layers
            num_layers = (num_layers, num_layers)
        except AttributeError:
            if hasattr(self._config, 'encoder_layers') and hasattr(self._config, 'decoder_layers'):
                num_layers = (self._config.encoder_layers, self._config.decoder_layers)
            else:
                raise AttributeError('could not find the number of encoder and decoder layers attributes in the model configuration, override the num_layers property of the model OnnxConfig to solve this')
        return num_layers

    @property
    def num_attention_heads(self) -> Tuple[int]:
        try:
            num_attention_heads = super().num_attention_heads
            num_attention_heads = (num_attention_heads, num_attention_heads)
        except AttributeError:
            if hasattr(self._config, 'encoder_attention_heads') and hasattr(self._config, 'decoder_attention_heads'):
                num_attention_heads = (self._config.encoder_attention_heads, self._config.decoder_attention_heads)
            else:
                raise AttributeError('could not find the number of attention heads for the encoder and the decoder attributes in the model configuration, override the num_attention_heads property of the model OnnxConfig to solve this')
        return num_attention_heads

    def generate_dummy_inputs(self, tokenizer: 'PreTrainedTokenizerBase', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
        encoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=decoder_seq_length, is_pair=is_pair, framework=framework)
        decoder_inputs = {f'decoder_{name}': tensor for name, tensor in decoder_inputs.items()}
        common_inputs = dict(**encoder_inputs, **decoder_inputs)
        if self.use_past:
            if not is_torch_available():
                raise ValueError('Cannot generate dummy past_keys inputs without PyTorch installed.')
            else:
                import torch
            batch = common_inputs['input_ids'].shape[0]
            encoder_seq_length = common_inputs['input_ids'].shape[1]
            decoder_seq_length = common_inputs['decoder_input_ids'].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            encoder_shape = (batch, num_encoder_attention_heads, encoder_seq_length, self._config.hidden_size // num_encoder_attention_heads)
            decoder_shape = (batch, num_decoder_attention_heads, decoder_seq_length + 3, self._config.hidden_size // num_decoder_attention_heads)
            common_inputs['past_key_values'] = []
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = 'encoder' if num_encoder_layers > num_decoder_layers else 'decoder'
            for _ in range(min_num_layers):
                common_inputs['past_key_values'].append((torch.zeros(decoder_shape), torch.zeros(decoder_shape), torch.zeros(encoder_shape), torch.zeros(encoder_shape)))
            shape = encoder_shape if remaining_side_name == 'encoder' else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs['past_key_values'].append((torch.zeros(shape), torch.zeros(shape)))
        return common_inputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        if direction not in ['inputs', 'outputs']:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
        name = 'past_key_values' if direction == 'inputs' else 'present'
        num_encoder_layers, num_decoder_layers = self.num_layers
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        remaining_side_name = 'encoder' if num_encoder_layers > num_decoder_layers else 'decoder'
        encoder_sequence = 'past_encoder_sequence'
        decoder_sequence = 'past_decoder_sequence' if direction == 'inputs' else 'past_decoder_sequence + sequence'
        for i in range(min_num_layers):
            inputs_or_outputs[f'{name}.{i}.decoder.key'] = {0: 'batch', 2: decoder_sequence}
            inputs_or_outputs[f'{name}.{i}.decoder.value'] = {0: 'batch', 2: decoder_sequence}
            inputs_or_outputs[f'{name}.{i}.encoder.key'] = {0: 'batch', 2: encoder_sequence}
            inputs_or_outputs[f'{name}.{i}.encoder.value'] = {0: 'batch', 2: encoder_sequence}
        for i in range(min_num_layers, max_num_layers):
            if remaining_side_name == 'encoder':
                axes_info = {0: 'batch', 2: encoder_sequence}
            else:
                axes_info = {0: 'batch', 2: decoder_sequence}
            inputs_or_outputs[f'{name}.{i}.{remaining_side_name}.key'] = axes_info

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f'{name}.{idx}.decoder.key'] = t[0]
        flattened_output[f'{name}.{idx}.decoder.value'] = t[1]
        flattened_output[f'{name}.{idx}.encoder.key'] = t[2]
        flattened_output[f'{name}.{idx}.encoder.value'] = t[3]