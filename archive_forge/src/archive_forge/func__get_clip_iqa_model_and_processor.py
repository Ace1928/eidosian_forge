from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _get_clip_iqa_model_and_processor(model_name_or_path: Literal['clip_iqa', 'openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']) -> Tuple['_CLIPModel', '_CLIPProcessor']:
    """Extract the CLIP model and processor from the model name or path."""
    from transformers import CLIPProcessor as _CLIPProcessor
    if model_name_or_path == 'clip_iqa':
        if not _PIQ_GREATER_EQUAL_0_8:
            raise ValueError("For metric `clip_iqa` to work with argument `model_name_or_path` set to default value `'clip_iqa'`, package `piq` version v0.8.0 or later must be installed. Either install with `pip install piq` or`pip install torchmetrics[multimodal]`")
        import piq
        model = piq.clip_iqa.clip.load().eval()
        processor = _CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        return (model, processor)
    return _get_clip_model_and_processor(model_name_or_path)