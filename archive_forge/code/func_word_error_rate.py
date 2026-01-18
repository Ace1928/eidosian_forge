from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def word_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Word error rate (WordErrorRate_) is a common metric of performance of an automatic speech recognition system.

    This value indicates the percentage of words that were incorrectly predicted. The lower the value, the better the
    performance of the ASR system with a WER of 0 being a perfect score.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Word error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_error_rate(preds=preds, target=target)
        tensor(0.5000)

    """
    errors, total = _wer_update(preds, target)
    return _wer_compute(errors, total)