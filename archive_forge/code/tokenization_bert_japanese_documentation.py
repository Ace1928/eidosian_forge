import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging

        Tokenizes text by sentencepiece. Based on [SentencePiece](https://github.com/google/sentencepiece).
        Tokenization needs the given vocabulary.

        Args:
            text: A string needs to be tokenized.

        Returns:
            A list of sentencepiece tokens.
        