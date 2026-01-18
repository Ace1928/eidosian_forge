import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_prophetnet import ProphetNetConfig

        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ProphetNetForCausalLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        >>> model = ProphetNetForCausalLM.from_pretrained("microsoft/prophetnet-large-uncased")
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits

        >>> # Model can also be used with EncoderDecoder framework
        >>> from transformers import BertTokenizer, EncoderDecoderModel, AutoTokenizer
        >>> import torch

        >>> tokenizer_enc = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
        >>> tokenizer_dec = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google-bert/bert-large-uncased", "microsoft/prophetnet-large-uncased"
        ... )

        >>> ARTICLE = (
        ...     "the us state department said wednesday it had received no "
        ...     "formal word from bolivia that it was expelling the us ambassador there "
        ...     "but said the charges made against him are `` baseless ."
        ... )
        >>> input_ids = tokenizer_enc(ARTICLE, return_tensors="pt").input_ids
        >>> labels = tokenizer_dec(
        ...     "us rejects charges against its ambassador in bolivia", return_tensors="pt"
        ... ).input_ids
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-1], labels=labels[:, 1:])

        >>> loss = outputs.loss
        ```