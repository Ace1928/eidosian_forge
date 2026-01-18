import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig

        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> image_urls = [
        ...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ... ]
        >>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        >>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        >>> inputs = processor(images, texts, padding=True, return_tensors="pt")
        >>> loss = model(**inputs, return_loss=True).loss

        >>> inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
        >>> loss_swapped = model(**inputs, return_loss=True).loss

        >>> print("Loss", round(loss.item(), 4))
        Loss 0.0019

        >>> print("Loss with swapped images", round(loss_swapped.item(), 4))
        Loss with swapped images 2.126
        ```