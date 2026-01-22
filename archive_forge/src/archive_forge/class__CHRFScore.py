from typing import Any, Literal, Optional, Sequence
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.text.eed import ExtendedEditDistance
from torchmetrics.text.mer import MatchErrorRate
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from torchmetrics.text.squad import SQuAD
from torchmetrics.text.ter import TranslationEditRate
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.wil import WordInfoLost
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _CHRFScore(CHRFScore):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> chrf = _CHRFScore()
    >>> chrf(preds, target)
    tensor(0.8640)

    """

    def __init__(self, n_char_order: int=6, n_word_order: int=2, beta: float=2.0, lowercase: bool=False, whitespace: bool=False, return_sentence_level_score: bool=False, **kwargs: Any) -> None:
        _deprecated_root_import_class('CHRFScore', 'text')
        super().__init__(n_char_order=n_char_order, n_word_order=n_word_order, beta=beta, lowercase=lowercase, whitespace=whitespace, return_sentence_level_score=return_sentence_level_score, **kwargs)