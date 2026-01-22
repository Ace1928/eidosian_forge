from typing import Any, Optional
from torchmetrics.retrieval.average_precision import RetrievalMAP
from torchmetrics.retrieval.fall_out import RetrievalFallOut
from torchmetrics.retrieval.hit_rate import RetrievalHitRate
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from torchmetrics.retrieval.precision import RetrievalPrecision
from torchmetrics.retrieval.precision_recall_curve import RetrievalPrecisionRecallCurve, RetrievalRecallAtFixedPrecision
from torchmetrics.retrieval.r_precision import RetrievalRPrecision
from torchmetrics.retrieval.recall import RetrievalRecall
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _RetrievalHitRate(RetrievalHitRate):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> indexes = tensor([0, 0, 0, 1, 1, 1, 1])
    >>> preds = tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
    >>> target = tensor([True, False, False, False, True, False, True])
    >>> hr2 = _RetrievalHitRate(top_k=2)
    >>> hr2(preds, target, indexes=indexes)
    tensor(0.5000)

    """

    def __init__(self, empty_target_action: str='neg', ignore_index: Optional[int]=None, top_k: Optional[int]=None, **kwargs: Any) -> None:
        _deprecated_root_import_class('RetrievalHitRate', 'retrieval')
        super().__init__(empty_target_action=empty_target_action, ignore_index=ignore_index, top_k=top_k, **kwargs)