import os
import re
import tempfile
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Sequence, Type
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import (
@classmethod
def _tokenize_zh(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenization of Chinese text.

        This is done in two steps: separate each Chinese characters (by utf-8 encoding) and afterwards tokenize the
        Chinese part (following the `13a` i.e. mteval tokenizer).
        Author: Shujian Huang huangsj@nju.edu.cn.

        Args:
            line: input sentence

        Return:
            tokenized sentence

        """
    line = line.strip()
    line_in_chars = ''
    for char in line:
        if cls._is_chinese_char(char):
            line_in_chars += ' '
            line_in_chars += char
            line_in_chars += ' '
        else:
            line_in_chars += char
    return cls._tokenize_regex(line_in_chars)