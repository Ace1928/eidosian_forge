from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig

        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth image matting for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        ... )
        >>> trimap = Image.open(filepath).convert("L")

        >>> # prepare image + trimap for the model
        >>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

        >>> with torch.no_grad():
        ...     alphas = model(**inputs).alphas
        >>> print(alphas.shape)
        torch.Size([1, 1, 640, 960])
        ```