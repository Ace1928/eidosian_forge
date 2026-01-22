import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _eed_update(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], language: Literal['en', 'ja']='en', alpha: float=2.0, rho: float=0.3, deletion: float=0.2, insertion: float=1.0, sentence_eed: Optional[List[Tensor]]=None) -> List[Tensor]:
    """Compute scores for ExtendedEditDistance.

    Args:
        preds: An iterable of hypothesis corpus
        target: An iterable of iterables of reference corpus
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character
        sentence_eed: list of sentence-level scores

    Return:
        individual sentence scores as a list of Tensors

    """
    preds, target = _preprocess_sentences(preds, target, language)
    if sentence_eed is None:
        sentence_eed = []
    if 0 in (len(preds), len(target[0])):
        return sentence_eed
    for hypothesis, target_words in zip(preds, target):
        score = _compute_sentence_statistics(hypothesis, target_words, alpha, rho, deletion, insertion)
        sentence_eed.append(score)
    return sentence_eed