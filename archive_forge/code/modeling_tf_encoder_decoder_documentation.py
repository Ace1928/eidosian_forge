from __future__ import annotations
import inspect
import re
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig

        Returns:

        Examples:

        ```python
        >>> from transformers import TFEncoderDecoderModel, BertTokenizer

        >>> # initialize a bert2gpt2 from a pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "openai-community/gpt2")

        >>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")

        >>> # forward
        >>> input_ids = tokenizer.encode(
        ...     "Hello, my dog is cute", add_special_tokens=True, return_tensors="tf"
        ... )  # Batch size 1
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

        >>> # training
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("bert2gpt2")
        >>> model = TFEncoderDecoderModel.from_pretrained("bert2gpt2")

        >>> # generation
        >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token_id)
        ```