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
class _TranslationEditRate(TranslationEditRate):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> ter = _TranslationEditRate()
    >>> ter(preds, target)
    tensor(0.1538)

    """

    def __init__(self, normalize: bool=False, no_punctuation: bool=False, lowercase: bool=True, asian_support: bool=False, return_sentence_level_score: bool=False, **kwargs: Any) -> None:
        _deprecated_root_import_class('TranslationEditRate', 'text')
        super().__init__(normalize=normalize, no_punctuation=no_punctuation, lowercase=lowercase, asian_support=asian_support, return_sentence_level_score=return_sentence_level_score, **kwargs)