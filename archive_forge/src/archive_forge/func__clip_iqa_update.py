from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _clip_iqa_update(model_name_or_path: str, images: Tensor, model: '_CLIPModel', processor: '_CLIPProcessor', data_range: float, device: Union[str, torch.device]) -> Tensor:
    images = images / float(data_range)
    'Update function for CLIP IQA.'
    if model_name_or_path == 'clip_iqa':
        default_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        default_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        images = (images - default_mean) / default_std
        img_features = model.encode_image(images.float(), pos_embedding=False).float()
    else:
        processed_input = processor(images=[i.cpu() for i in images], return_tensors='pt', padding=True)
        img_features = model.get_image_features(processed_input['pixel_values'].to(device))
    return img_features / img_features.norm(p=2, dim=-1, keepdim=True)