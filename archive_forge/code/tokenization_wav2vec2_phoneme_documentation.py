import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (

        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better
                understand how to make use of `output_word_offsets`.
                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works analogous with phonemes
                and batched output.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The
            decoded sentence. Will be a
            [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`] when
            `output_char_offsets == True`.
        