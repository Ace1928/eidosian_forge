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
class _ExtendedEditDistance(ExtendedEditDistance):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "here is an other sample"]
    >>> target = ["this is the reference", "here is another one"]
    >>> eed = _ExtendedEditDistance()
    >>> eed(preds=preds, target=target)
    tensor(0.3078)

    """

    def __init__(self, language: Literal['en', 'ja']='en', return_sentence_level_score: bool=False, alpha: float=2.0, rho: float=0.3, deletion: float=0.2, insertion: float=1.0, **kwargs: Any) -> None:
        _deprecated_root_import_class('ExtendedEditDistance', 'text')
        super().__init__(language=language, return_sentence_level_score=return_sentence_level_score, alpha=alpha, rho=rho, deletion=deletion, insertion=insertion, **kwargs)