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
def _tokenize_regex(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Post-processing tokenizer for `13a` and `zh` tokenizers.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line

        """
    for _re, repl in cls._REGEX:
        line = _re.sub(repl, line)
    return ' '.join(line.split())