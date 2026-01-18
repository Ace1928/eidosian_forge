import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig

        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:

        Examples:

        Semantic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> list(predicted_semantic_map.shape)
        [512, 683]
        ```

        Panoptic segmentation example:

        ```python
        >>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on COCO panoptic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        >>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to image_processor for postprocessing
        >>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        >>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        >>> predicted_panoptic_map = result["segmentation"]
        >>> list(predicted_panoptic_map.shape)
        [480, 640]
        ```
        