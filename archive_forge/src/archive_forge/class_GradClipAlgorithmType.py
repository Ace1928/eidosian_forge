from __future__ import annotations
from lightning_utilities.core.enums import StrEnum as LightningEnum
class GradClipAlgorithmType(LightningEnum):
    """Define gradient_clip_algorithm types - training-tricks.
    NORM type means "clipping gradients by norm". This computed over all model parameters together.
    VALUE type means "clipping gradients by value". This will clip the gradient value for each parameter.

    References:
        clip_by_norm: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_by_value: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_value_
    """
    VALUE = 'value'
    NORM = 'norm'

    @staticmethod
    def supported_type(val: str) -> bool:
        return any((x.value == val for x in GradClipAlgorithmType))

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in GradClipAlgorithmType]