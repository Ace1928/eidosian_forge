import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _preprocess_en(sentence: str) -> str:
    """Preprocess english sentences.

    Copied from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/util.py.

    Raises:
        ValueError: If input sentence is not of a type `str`.

    """
    if not isinstance(sentence, str):
        raise ValueError(f'Only strings allowed during preprocessing step, found {type(sentence)} instead')
    sentence = sentence.rstrip()
    rules_interpunction = [('.', ' .'), ('!', ' !'), ('?', ' ?'), (',', ' ,')]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)
    rules_re = [('\\s+', ' '), ('(\\d) ([.,]) (\\d)', '\\1\\2\\3'), ('(Dr|Jr|Prof|Rev|Gen|Mr|Mt|Mrs|Ms) .', '\\1.')]
    for pattern, replacement in rules_re:
        sentence = re.sub(pattern, replacement, sentence)
    rules_interpunction = [('e . g .', 'e.g.'), ('i . e .', 'i.e.'), ('U . S .', 'U.S.')]
    for pattern, replacement in rules_interpunction:
        sentence = sentence.replace(pattern, replacement)
    return ' ' + sentence + ' '