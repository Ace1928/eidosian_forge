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
class _Perplexity(Perplexity):
    """Wrapper for deprecated import.

    >>> import torch
    >>> gen = torch.manual_seed(42)
    >>> preds = torch.rand(2, 8, 5, generator=gen)
    >>> target = torch.randint(5, (2, 8), generator=gen)
    >>> target[0, 6:] = -100
    >>> perp = _Perplexity(ignore_index=-100)
    >>> perp(preds, target)
    tensor(5.8540)

    """

    def __init__(self, ignore_index: Optional[int]=None, **kwargs: Any) -> None:
        _deprecated_root_import_class('Perplexity', 'text')
        super().__init__(ignore_index=ignore_index, **kwargs)