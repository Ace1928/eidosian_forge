import math
import os
import warnings
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_imagegpt import ImageGPTConfig
@add_start_docstrings('\n    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', IMAGEGPT_START_DOCSTRING)
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)
        self.transformer = ImageGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[bool]=None, **kwargs):
        token_type_ids = kwargs.get('token_type_ids', None)
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs: Any) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
        >>> import torch
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
        >>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> # unconditional generation of 8 images
        >>> batch_size = 4
        >>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  # initialize with SOS token
        >>> context = context.to(device)
        >>> output = model.generate(
        ...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
        ... )

        >>> clusters = image_processor.clusters
        >>> height = image_processor.size["height"]
        >>> width = image_processor.size["width"]

        >>> samples = output[:, 1:].cpu().detach().numpy()
        >>> samples_img = [
        ...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
        ... ]  # convert color cluster tokens back to pixels
        >>> f, axes = plt.subplots(1, batch_size, dpi=300)

        >>> for img, ax in zip(samples_img, axes):  # doctest: +IGNORE_RESULT
        ...     ax.axis("off")
        ...     ax.imshow(img)
        ```"""
        if 'pixel_values' in kwargs:
            warnings.warn('The `pixel_values` argument is deprecated and will be removed in a future version, use `input_ids` instead.', FutureWarning)
            if input_ids is not None:
                raise ValueError('You cannot pass both `pixel_values` and `input_ids`. Please make sure to only pass `input_ids`.')
            input_ids = kwargs.pop('pixel_values')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions, cross_attentions=transformer_outputs.cross_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))