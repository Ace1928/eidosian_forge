from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate, e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`tf.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If unset the method
                initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should of in
                the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`,
                `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            seed (`List[int]`, *optional*):
                Random seed to control sampling, containing two integers, used when `do_sample` is `True`. See the
                `seed` argument from stateless functions in `tf.random`.
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `TFWhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`tf.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `tf.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True` or when
            `config.return_dict_in_generate=True`) or a `tf.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchDecoderOnlyOutput`],
                    - [`~generation.TFSampleDecoderOnlyOutput`],
                    - [`~generation.TFBeamSearchDecoderOnlyOutput`],
                    - [`~generation.TFBeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.TFGreedySearchEncoderDecoderOutput`],
                    - [`~generation.TFSampleEncoderDecoderOutput`],
                    - [`~generation.TFBeamSearchEncoderDecoderOutput`],
                    - [`~generation.TFBeamSampleEncoderDecoderOutput`]

        