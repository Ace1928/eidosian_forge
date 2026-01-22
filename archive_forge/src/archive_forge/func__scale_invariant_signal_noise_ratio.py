from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> _scale_invariant_signal_noise_ratio(preds, target)
    tensor(15.0918)

    """
    _deprecated_root_import_func('scale_invariant_signal_noise_ratio', 'audio')
    return scale_invariant_signal_noise_ratio(preds=preds, target=target)