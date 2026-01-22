import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging
class SudachiTokenizer:
    """Runs basic tokenization with Sudachi morphological parser."""

    def __init__(self, do_lower_case=False, never_split=None, normalize_text=True, trim_whitespace=False, sudachi_split_mode='A', sudachi_config_path=None, sudachi_resource_dir=None, sudachi_dict_type='core', sudachi_projection=None):
        """
        Constructs a SudachiTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **trim_whitespace**: (*optional*) boolean (default False)
                Whether to trim all whitespace, tab, newline from tokens.
            **sudachi_split_mode**: (*optional*) string
                Split mode of sudachi, choose from `["A", "B", "C"]`.
            **sudachi_config_path**: (*optional*) string
            **sudachi_resource_dir**: (*optional*) string
            **sudachi_dict_type**: (*optional*) string
                dict type of sudachi, choose from `["small", "core", "full"]`.
            **sudachi_projection**: (*optional*) string
                Word projection mode of sudachi, choose from `["surface", "normalized", "reading", "dictionary", "dictionary_and_surface", "normalized_and_surface", "normalized_nouns"]`.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text
        self.trim_whitespace = trim_whitespace
        try:
            from sudachipy import dictionary, tokenizer
        except ImportError:
            raise ImportError('You need to install sudachipy to use SudachiTokenizer. See https://github.com/WorksApplications/SudachiPy for installation.')
        if sudachi_split_mode == 'A':
            self.split_mode = tokenizer.Tokenizer.SplitMode.A
        elif sudachi_split_mode == 'B':
            self.split_mode = tokenizer.Tokenizer.SplitMode.B
        elif sudachi_split_mode == 'C':
            self.split_mode = tokenizer.Tokenizer.SplitMode.C
        else:
            raise ValueError('Invalid sudachi_split_mode is specified.')
        self.projection = sudachi_projection
        sudachi_dictionary = dictionary.Dictionary(config_path=sudachi_config_path, resource_dir=sudachi_resource_dir, dict=sudachi_dict_type)
        if is_sudachi_projection_available():
            self.sudachi = sudachi_dictionary.create(self.split_mode, projection=self.projection)
        elif self.projection is not None:
            raise ImportError('You need to install sudachipy>=0.6.8 to specify `projection` field in sudachi_kwargs.')
        else:
            self.sudachi = sudachi_dictionary.create(self.split_mode)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize('NFKC', text)
        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        for word in self.sudachi.tokenize(text):
            token = word.surface()
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            if self.trim_whitespace:
                if token.strip() == '':
                    continue
                else:
                    token = token.strip()
            tokens.append(token)
        return tokens