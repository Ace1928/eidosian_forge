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
def _tokenize_ko_mecab(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenizes a Korean string line using MeCab-korean morphological analyzer.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
    import mecab_ko
    import mecab_ko_dic
    tagger = mecab_ko.Tagger(mecab_ko_dic.MECAB_ARGS + ' -Owakati')
    line = line.strip()
    return tagger.parse(line).strip()