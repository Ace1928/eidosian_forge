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
class _MatchErrorRate(MatchErrorRate):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> mer = _MatchErrorRate()
    >>> mer(preds, target)
    tensor(0.4444)

    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class('MatchErrorRate', 'text')
        super().__init__(**kwargs)