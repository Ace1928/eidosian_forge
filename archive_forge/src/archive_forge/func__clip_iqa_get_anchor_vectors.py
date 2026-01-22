from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _clip_iqa_get_anchor_vectors(model_name_or_path: str, model: '_CLIPModel', processor: '_CLIPProcessor', prompts_list: List[str], device: Union[str, torch.device]) -> Tensor:
    """Calculates the anchor vectors for the CLIP IQA metric.

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use.
        model: The CLIP model
        processor: The CLIP processor
        prompts_list: A list of prompts
        device: The device to use for the calculation

    """
    if model_name_or_path == 'clip_iqa':
        text_processed = processor(text=prompts_list)
        anchors_text = torch.zeros(len(prompts_list), processor.tokenizer.model_max_length, dtype=torch.long, device=device)
        for i, tp in enumerate(text_processed['input_ids']):
            anchors_text[i, :len(tp)] = torch.tensor(tp, dtype=torch.long, device=device)
        anchors = model.encode_text(anchors_text).float()
    else:
        text_processed = processor(text=prompts_list, return_tensors='pt', padding=True)
        anchors = model.get_text_features(text_processed['input_ids'].to(device), text_processed['attention_mask'].to(device))
    return anchors / anchors.norm(p=2, dim=-1, keepdim=True)