import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (

        Generate method for `ClvpModelForConditionalGeneration`, this method calls the `generate` method of
        `ClvpForCausalLM` and then uses those generated `speech_ids` to process `text_embeds` and `speech_embeds` using
        `ClvpEncoder`.

        Args:
            input_ids (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input text Tokens. Processed from the [`ClvpTokenizer`].
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`, *optional*):
                Indicates log-melspectrogram representations for audio returned by [`ClvpFeatureExtractor`].
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            pad_to_max_mel_tokens (`int`, *optional*):
                Pads generated speech_ids to the specified value. This is to implement the same logic from the official
                repo, link: https://github.com/neonbjb/tortoise-tts/blob/80f89987a5abda5e2b082618cd74f9c7411141dc/tortoise/api.py#L430
                and to make sure the logits are same.
                This does not affect generation quality so please don't consider using it since it is less efficient.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of decoder model, text encoder and speech encoder models.

        Returns:
            `ClvpOutput` or tuple: A `ClvpOutput` (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a tuple.
        