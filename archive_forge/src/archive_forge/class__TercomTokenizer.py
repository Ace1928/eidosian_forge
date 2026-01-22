import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
class _TercomTokenizer:
    """Re-implementation of Tercom Tokenizer in Python 3.

    See src/ter/core/Normalizer.java in https://github.com/jhclark/tercom Note that Python doesn't support named Unicode
    blocks so the mapping for relevant blocks was taken from here: https://unicode-table.com/en/blocks/

    This implementation follows the implementation from
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_ter.py.

    """
    _ASIAN_PUNCTUATION = '([\\u3001\\u3002\\u3008-\\u3011\\u3014-\\u301f\\uff61-\\uff65\\u30fb])'
    _FULL_WIDTH_PUNCTUATION = '([\\uff0e\\uff0c\\uff1f\\uff1a\\uff1b\\uff01\\uff02\\uff08\\uff09])'

    def __init__(self, normalize: bool=False, no_punctuation: bool=False, lowercase: bool=True, asian_support: bool=False) -> None:
        """Initialize the tokenizer.

        Args:
            normalize: An indication whether a general tokenization to be applied.
            no_punctuation: An indication whteher a punctuation to be removed from the sentences.
            lowercase: An indication whether to enable case-insensitivity.
            asian_support: An indication whether asian characters to be processed.

        """
        self.normalize = normalize
        self.no_punctuation = no_punctuation
        self.lowercase = lowercase
        self.asian_support = asian_support

    @lru_cache(maxsize=2 ** 16)
    def __call__(self, sentence: str) -> str:
        """Apply a different tokenization techniques according.

        Args:
            sentence: An input sentence to pre-process and tokenize.

        Return:
            A tokenized and pre-processed sentence.

        """
        if not sentence:
            return ''
        if self.lowercase:
            sentence = sentence.lower()
        if self.normalize:
            sentence = self._normalize_general_and_western(sentence)
            if self.asian_support:
                sentence = self._normalize_asian(sentence)
        if self.no_punctuation:
            sentence = self._remove_punct(sentence)
            if self.asian_support:
                sentence = self._remove_asian_punct(sentence)
        return ' '.join(sentence.split())

    @staticmethod
    def _normalize_general_and_western(sentence: str) -> str:
        """Apply a language-independent (general) tokenization."""
        sentence = f' {sentence} '
        rules = [('\\n-', ''), ('\\n', ' '), ('&quot;', '"'), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('([{-~[-` -&(-+:-@/])', ' \\1 '), ("'s ", " 's "), ("'s$", " 's"), ('([^0-9])([\\.,])', '\\1 \\2 '), ('([\\.,])([^0-9])', ' \\1 \\2'), ('([0-9])(-)', '\\1 \\2 ')]
        for pattern, replacement in rules:
            sentence = re.sub(pattern, replacement, sentence)
        return sentence

    @classmethod
    def _normalize_asian(cls: Type['_TercomTokenizer'], sentence: str) -> str:
        """Split Chinese chars and Japanese kanji down to character level."""
        sentence = re.sub('([\\u4e00-\\u9fff\\u3400-\\u4dbf])', ' \\1 ', sentence)
        sentence = re.sub('([\\u31c0-\\u31ef\\u2e80-\\u2eff])', ' \\1 ', sentence)
        sentence = re.sub('([\\u3300-\\u33ff\\uf900-\\ufaff\\ufe30-\\ufe4f])', ' \\1 ', sentence)
        sentence = re.sub('([\\u3200-\\u3f22])', ' \\1 ', sentence)
        sentence = re.sub('(^|^[\\u3040-\\u309f])([\\u3040-\\u309f]+)(?=$|^[\\u3040-\\u309f])', '\\1 \\2 ', sentence)
        sentence = re.sub('(^|^[\\u30a0-\\u30ff])([\\u30a0-\\u30ff]+)(?=$|^[\\u30a0-\\u30ff])', '\\1 \\2 ', sentence)
        sentence = re.sub('(^|^[\\u31f0-\\u31ff])([\\u31f0-\\u31ff]+)(?=$|^[\\u31f0-\\u31ff])', '\\1 \\2 ', sentence)
        sentence = re.sub(cls._ASIAN_PUNCTUATION, ' \\1 ', sentence)
        return re.sub(cls._FULL_WIDTH_PUNCTUATION, ' \\1 ', sentence)

    @staticmethod
    def _remove_punct(sentence: str) -> str:
        """Remove punctuation from an input sentence string."""
        return re.sub('[\\.,\\?:;!\\"\\(\\)]', '', sentence)

    @classmethod
    def _remove_asian_punct(cls: Type['_TercomTokenizer'], sentence: str) -> str:
        """Remove asian punctuation from an input sentence string."""
        sentence = re.sub(cls._ASIAN_PUNCTUATION, '', sentence)
        return re.sub(cls._FULL_WIDTH_PUNCTUATION, '', sentence)