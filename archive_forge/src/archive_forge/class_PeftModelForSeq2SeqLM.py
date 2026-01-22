from __future__ import annotations
import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from . import __version__
from .config import PeftConfig
from .tuners import (
from .utils import (
class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import PeftModelForSeq2SeqLM, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "SEQ_2_SEQ_LM",
        ...     "inference_mode": False,
        ...     "r": 8,
        ...     "target_modules": ["q", "v"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.1,
        ...     "fan_in_fan_out": False,
        ...     "enable_lora": None,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str='default') -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = self.base_model._prepare_encoder_decoder_kwargs_for_generation

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, decoder_input_ids=None, decoder_attention_mask=None, decoder_inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, task_ids=None, **kwargs):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs['task_ids'] = task_ids
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_inputs_embeds=decoder_inputs_embeds, labels=labels, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if decoder_attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(decoder_attention_mask.device)
            if peft_config.peft_type not in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)
        if kwargs.get('position_ids', None) is not None:
            warnings.warn('Position ids are not supported for parameter efficient tuning. Ignoring position ids.')
            kwargs['position_ids'] = None
        if kwargs.get('token_type_ids', None) is not None:
            warnings.warn('Token type ids are not supported for parameter efficient tuning. Ignoring token type ids')
            kwargs['token_type_ids'] = None
        kwargs.update({'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'labels': labels, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states, 'return_dict': return_dict})
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, decoder_inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, **kwargs)
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
                kwargs['attention_mask'] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, :peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, decoder_input_ids=decoder_input_ids, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
                kwargs['attention_mask'] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs['labels'] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs['labels'] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, :peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat((prompts[:, peft_config.num_virtual_tokens:], decoder_inputs_embeds), dim=1)
                return self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = self._prepare_encoder_decoder_kwargs_for_generation
        try:
            if not peft_config.is_prompt_learning:
                outputs = self.base_model.generate(**kwargs)
            else:
                if 'input_ids' not in kwargs:
                    raise ValueError('input_ids must be provided for Peft model generation')
                if kwargs.get('position_ids', None) is not None:
                    warnings.warn('Position ids are not supported for parameter efficient tuning. Ignoring position ids.')
                    kwargs['position_ids'] = None
                if kwargs.get('token_type_ids', None) is not None:
                    warnings.warn('Token type ids are not supported for parameter efficient tuning. Ignoring token type ids')
                    kwargs['token_type_ids'] = None
                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING, PeftType.MULTITASK_PROMPT_TUNING]:
                    kwargs = deepcopy(kwargs)
                    if 'encoder_outputs' in kwargs:
                        del kwargs['encoder_outputs']
                        warnings.warn('`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it.')
                    input_ids = kwargs.pop('input_ids')
                    inputs_embeds = self.word_embeddings(input_ids)
                    batch_size = inputs_embeds.shape[0]
                    prompts = self.get_prompt(batch_size=batch_size, task_ids=kwargs.pop('task_ids', None))
                    prompts = prompts.to(inputs_embeds.dtype)
                    inputs_embeds = torch.cat((prompts[:, :peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                    kwargs['inputs_embeds'] = inputs_embeds
                    if 'attention_mask' in kwargs:
                        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(kwargs['attention_mask'].device)
                        kwargs['attention_mask'] = torch.cat((prefix_attention_mask, kwargs['attention_mask']), dim=1)
                    return self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = self.base_model_prepare_encoder_decoder_kwargs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = self.base_model_prepare_encoder_decoder_kwargs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor=None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs['task_ids'] = task_ids
        if model_kwargs['past_key_values'] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs['decoder_input_ids'].shape[0]
            past_key_values = self.get_prompt(batch_size)
            model_kwargs['past_key_values'] = past_key_values
        return model_kwargs