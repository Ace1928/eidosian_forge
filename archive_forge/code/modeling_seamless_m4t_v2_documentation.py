import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config

        Generates translated token ids and/or translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
        perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>


        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*):
                Input audio features. This should be returnes by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
                ignored.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            speaker_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            generate_speech (`bool`, *optional*, defaults to `True`):
                If `False`, will only returns the text tokens and won't generate speech.

            kwargs (*optional*):
                Remaining dictioy of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.

        Returns:
            `Union[SeamlessM4Tv2GenerationOutput, Tuple[Tensor], ModelOutput]`:
            - If `generate_speech` and `return_intermediate_token_ids`, returns [`SeamlessM4Tv2GenerationOutput`].
            - If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
              shape `(batch_size, sequence_length)`and and `waveform_lengths` which gives the length of each sample.
            - If `generate_speech=False`, it will returns `ModelOutput`.
        