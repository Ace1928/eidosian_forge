import os
from typing import Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig

        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.

        Params:
            encoder_pretrained_model_name_or_path (`Union[str, os.PathLike]`, *optional*):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`Union[str, os.PathLike]`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import FlaxEncoderDecoderModel

        >>> # initialize a bert2gpt2 from pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "openai-community/gpt2")
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./bert2gpt2")
        >>> # load fine-tuned model
        >>> model = FlaxEncoderDecoderModel.from_pretrained("./bert2gpt2")
        ```