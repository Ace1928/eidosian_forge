from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from onnxruntime import InferenceSession
from ..utils import NormalizedConfigManager
from ..utils.logging import warn_once
from .utils import get_ordered_input_names, logging
class ORTDecoderForSeq2Seq(ORTModelPart):
    """
    Decoder model with a language modeling head on top for ONNX Runtime inference.
    """

    def __init__(self, session: InferenceSession, parent_model: 'ORTModel'):
        super().__init__(session, parent_model)
        self.key_value_input_names = [key for key in self.input_names if '.key' in key or '.value' in key]
        self.key_value_output_names = [key for key in self.output_names if '.key' in key or '.value' in key]
        if len(self.key_value_input_names) == 0:
            self.key_value_input_names = [key for key in self.input_names if 'key_values' in key]
        if len(self.key_value_output_names) == 0:
            self.key_value_output_names = [key for key in self.output_names if 'key_values' in key]
        if self.parent_model.use_cache is True and len(self.key_value_output_names) == 0:
            raise RuntimeError('Could not find the past key values in the provided model.')
        self.use_past_in_outputs = len(self.key_value_output_names) > 0
        self.use_past_in_inputs = len(self.key_value_input_names) > 0
        self.use_fp16 = False
        for inp in session.get_inputs():
            if 'past_key_values' in inp.name and inp.type == 'tensor(float16)':
                self.use_fp16 = True
                break
        self.no_cross_attention_cache = getattr(self.parent_model, 'no_cross_attention_cache', False)
        if not self.parent_model.use_merged and self.use_past_in_inputs or self.no_cross_attention_cache:
            self.num_pkv = 2
        else:
            self.num_pkv = 4
        self.past_key_values_cross_attention_output_names = set()
        for output_name in self.output_names:
            if output_name.startswith('present') and 'encoder' in output_name:
                self.past_key_values_cross_attention_output_names.add(output_name)
        self.use_legacy_outputs = self.parent_model.use_merged is False and len(self.past_key_values_cross_attention_output_names) > 0

    def compute_past_key_values_output_shapes(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, use_cache_branch: Optional[bool], past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None) -> Dict[str, int]:
        batch_size = input_ids.size(0)
        num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads
        sequence_length = input_ids.size(1)
        encoder_sequence_length = encoder_hidden_states.size(1)
        if past_key_values is not None and use_cache_branch is not False:
            sequence_length += past_key_values[0].size(2)
        self_attn_shape = (batch_size, num_attention_heads, sequence_length, embed_size_per_head)
        if past_key_values is not None and use_cache_branch is True:
            cross_attn_shape = (0, num_attention_heads, 1, embed_size_per_head)
        else:
            cross_attn_shape = (batch_size, num_attention_heads, encoder_sequence_length, embed_size_per_head)
        past_key_values_shapes = {}
        for idx, name in enumerate(self.key_value_output_names):
            is_self_attn = idx % 4 < 2
            past_key_values_shapes[name] = self_attn_shape if is_self_attn or self.num_pkv == 2 else cross_attn_shape
        return past_key_values_shapes

    def get_outputs_not_to_bind(self, use_merged_cache: bool) -> Set[str]:
        result = {output_name for output_name in self.output_names if not output_name.startswith('present') and output_name not in {'loss', 'logits'}}
        if use_merged_cache is True:
            result = result.union(self.past_key_values_cross_attention_output_names)
        return result

    def forward(self, input_ids: torch.LongTensor, encoder_hidden_states: torch.FloatTensor, decoder_attention_mask: Optional[torch.LongTensor]=None, encoder_attention_mask: Optional[torch.LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, use_cache_branch: None=None) -> Seq2SeqLMOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.parent_model.raise_on_numpy_input_io_binding(use_torch)
        if past_key_values is not None:
            past_key_values = tuple((past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer))
        use_merged_no_cache = past_key_values is None and self.parent_model.use_merged
        use_merged_cache = past_key_values is not None and self.parent_model.use_merged
        use_cache_branch_tensor, past_key_values = self.prepare_inputs_for_merged(input_ids, past_key_values, use_torch=use_torch)
        if self.parent_model.use_io_binding:
            known_output_shapes = self.compute_past_key_values_output_shapes(input_ids, encoder_hidden_states, use_cache_branch=use_cache_branch_tensor.item() if use_cache_branch_tensor is not None else None, past_key_values=past_key_values)
            outputs_to_not_bind = self.get_outputs_not_to_bind(use_merged_cache)
            model_inputs = [input_ids]
            if 'encoder_hidden_states' in self.input_names:
                model_inputs.append(encoder_hidden_states)
            if 'decoder_attention_mask' in self.input_names:
                model_inputs.append(decoder_attention_mask)
            if 'encoder_attention_mask' in self.input_names:
                model_inputs.append(encoder_attention_mask)
            if past_key_values is not None:
                model_inputs += past_key_values
            if 'labels' in self.input_names:
                model_inputs.append(labels)
                known_output_shapes.update({'loss': []})
            if use_cache_branch_tensor is not None:
                model_inputs.append(use_cache_branch_tensor)
            io_binding, output_shapes, output_buffers = self.parent_model._prepare_io_binding(self.session, *model_inputs, known_output_shapes=known_output_shapes, ordered_input_names=self._ordered_input_names, outputs_to_not_bind=outputs_to_not_bind)
            for name, shape in output_shapes.items():
                if name in self.key_value_output_names:
                    output_shapes[name] = shape[:2] + (-1,) + shape[3:]
            io_binding.synchronize_inputs()
            self.session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            out_past_key_values = ()
            for name in self.key_value_output_names:
                if name in self.past_key_values_cross_attention_output_names and use_merged_cache:
                    continue
                out_past_key_values += (output_buffers[name].view(output_shapes[name]),)
            logits = output_buffers['logits'].view(output_shapes['logits'])
            loss = None
            if 'loss' in self.output_names:
                loss = output_buffers['loss'].view(output_shapes['loss'])
            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache:
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.use_legacy_outputs is True:
                msg = 'For the decoder with past, using ONNX models outputting cross attention past key values is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model with optimum>=1.7.3.'
                warn_once(logger, msg=msg)
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.num_pkv == 2:
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] + past_key_values[2 * i + 2:2 * i + 2 + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.num_pkv == 4:
                out_past_key_values = tuple((out_past_key_values[i:i + 2] + past_key_values[2 * i + 2:2 * i + 4] for i in range(0, len(out_past_key_values), 2)))
            else:
                raise ValueError('Unsupported num_pkv')
        else:
            if use_torch:
                onnx_inputs = {'input_ids': input_ids.cpu().detach().numpy()}
                if 'encoder_hidden_states' in self.input_names:
                    onnx_inputs['encoder_hidden_states'] = encoder_hidden_states.cpu().detach().numpy()
                if 'decoder_attention_mask' in self.input_names:
                    onnx_inputs['decoder_attention_mask'] = decoder_attention_mask.cpu().detach().numpy()
                if 'encoder_attention_mask' in self.input_names:
                    onnx_inputs['encoder_attention_mask'] = encoder_attention_mask.cpu().detach().numpy()
                if past_key_values is not None:
                    for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                        onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()
                if 'labels' in self.input_names:
                    onnx_inputs['labels'] = labels.cpu().detach().numpy()
                if self.parent_model.use_merged is True:
                    onnx_inputs['use_cache_branch'] = use_cache_branch_tensor.cpu().detach().numpy()
            else:
                onnx_inputs = {'input_ids': input_ids}
                if 'encoder_hidden_states' in self.input_names:
                    onnx_inputs['encoder_hidden_states'] = encoder_hidden_states
                if 'decoder_attention_mask' in self.input_names:
                    onnx_inputs['decoder_attention_mask'] = decoder_attention_mask
                if 'encoder_attention_mask' in self.input_names:
                    onnx_inputs['encoder_attention_mask'] = encoder_attention_mask
                if past_key_values is not None:
                    for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                        onnx_inputs[input_name] = past_key_value
                if 'labels' in self.input_names:
                    onnx_inputs['labels'] = labels
                if self.parent_model.use_merged is True:
                    onnx_inputs['use_cache_branch'] = use_cache_branch_tensor
            outputs = self.session.run(None, onnx_inputs)
            out_past_key_values = tuple((torch.from_numpy(outputs[self.output_names[key]]).to(self.device) for key in self.key_value_output_names))
            logits = outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            loss = None
            if 'loss' in self.output_names:
                loss = outputs[self.output_names['loss']]
                if use_torch:
                    loss = torch.from_numpy(loss).to(self.device)
            if not self.use_past_in_outputs:
                out_past_key_values = None
            elif not self.use_past_in_inputs or use_merged_no_cache or self.no_cross_attention_cache:
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.use_legacy_outputs is True:
                msg = 'For the decoder with past, using ONNX models outputting cross attention past key values is deprecated and the support will be removed in optimum 2.0. We recommend exporting again the model with optimum>=1.7.3.'
                warn_once(logger, msg=msg)
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.num_pkv == 2:
                out_past_key_values = tuple((out_past_key_values[i:i + self.num_pkv] + past_key_values[2 * i + 2:2 * i + 2 + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)))
            elif self.num_pkv == 4:
                out_past_key_values = tuple((out_past_key_values[i:i + 2] + past_key_values[i + 2:i + 4] for i in range(0, len(out_past_key_values), self.num_pkv)))
            else:
                raise ValueError('Unsupported num_pkv')
        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=out_past_key_values)

    def prepare_inputs_for_merged(self, input_ids: Union[None, torch.LongTensor, np.ndarray], past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]], use_torch: bool):
        if self.parent_model.use_merged:
            constructor = torch if use_torch is True else np
            use_cache_branch = constructor.full((1,), past_key_values is not None)
        else:
            use_cache_branch = None
        if use_torch and use_cache_branch is not None:
            use_cache_branch = use_cache_branch.to(self.device)
        if self.parent_model.use_merged and past_key_values is None:
            batch_size = input_ids.shape[0]
            num_attention_heads = self.normalized_config.num_attention_heads
            embed_size_per_head = self.normalized_config.hidden_size // num_attention_heads
            dtype = constructor.float16 if self.use_fp16 else constructor.float32
            shape = (batch_size, num_attention_heads, 1, embed_size_per_head)
            key_or_value = constructor.zeros(shape, dtype=dtype)
            if use_torch is True:
                key_or_value = key_or_value.to(self.device)
            past_key_values = tuple((key_or_value for _ in range(len(self.key_value_input_names))))
        return (use_cache_branch, past_key_values)