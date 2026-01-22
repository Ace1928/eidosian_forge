import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
def _build_translation_inputs(self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
    """Used by translation pipeline, to prepare inputs for the generate function"""
    if src_lang is None or tgt_lang is None:
        raise ValueError('Translation requires a `src_lang` and a `tgt_lang` for this model')
    self.src_lang = src_lang
    inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
    if '__' not in tgt_lang:
        tgt_lang = f'__{tgt_lang}__'
    tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
    inputs['forced_bos_token_id'] = tgt_lang_id
    return inputs