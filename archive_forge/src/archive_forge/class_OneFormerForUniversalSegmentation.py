import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
@add_start_docstrings('OneFormer Model for instance, semantic and panoptic image segmentation.', ONEFORMER_START_DOCSTRING)
class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    main_input_name = ['pixel_values', 'task_inputs']

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.model = OneFormerModel(config)
        self.matcher = OneFormerHungarianMatcher(cost_class=config.class_weight, cost_dice=config.dice_weight, cost_mask=config.mask_weight, num_points=config.train_num_points)
        self.weight_dict: Dict[str, float] = {'loss_cross_entropy': config.class_weight, 'loss_mask': config.mask_weight, 'loss_dice': config.dice_weight, 'loss_contrastive': config.contrastive_weight}
        self.criterion = OneFormerLoss(num_classes=config.num_labels, matcher=self.matcher, weight_dict=self.weight_dict, eos_coef=config.no_object_weight, num_points=config.train_num_points, oversample_ratio=config.oversample_ratio, importance_sample_ratio=config.importance_sample_ratio, contrastive_temperature=config.contrastive_temperature)
        self.post_init()

    def get_loss_dict(self, masks_queries_logits: Tensor, class_queries_logits: Tensor, contrastive_queries_logits: Tensor, mask_labels: Tensor, class_labels: Tensor, text_queries: Tensor, auxiliary_predictions: Dict[str, Tensor], calculate_contrastive_loss: bool) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, contrastive_queries_logits=contrastive_queries_logits, mask_labels=mask_labels, class_labels=class_labels, text_queries=text_queries, auxiliary_predictions=auxiliary_predictions, calculate_contrastive_loss=calculate_contrastive_loss)
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight
        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, task_inputs: Tensor, text_inputs: Optional[Tensor]=None, mask_labels: Optional[List[Tensor]]=None, class_labels: Optional[List[Tensor]]=None, pixel_mask: Optional[Tensor]=None, output_auxiliary_logits: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> OneFormerForUniversalSegmentationOutput:
        """
        text_inputs (`List[torch.Tensor]`, *optional*):
            Tensor fof shape `(num_queries, sequence_length)` to be fed to a model
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `OneFormerUniversalSegmentationOutput`
        Example:

        Universal segmentation example:

        ```python
        >>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # load OneFormer fine-tuned on ADE20k for universal segmentation
        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # Semantic Segmentation
        >>> inputs = processor(image, ["semantic"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for semantic postprocessing
        >>> predicted_semantic_map = processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"
        '👉 Semantic Predictions Shape: [512, 683]'

        >>> # Instance Segmentation
        >>> inputs = processor(image, ["instance"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for instance postprocessing
        >>> predicted_instance_map = processor.post_process_instance_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"
        '👉 Instance Predictions Shape: [512, 683]'

        >>> # Panoptic Segmentation
        >>> inputs = processor(image, ["panoptic"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for panoptic postprocessing
        >>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"
        '👉 Panoptic Predictions Shape: [512, 683]'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(pixel_values=pixel_values, task_inputs=task_inputs, text_inputs=text_inputs, pixel_mask=pixel_mask, output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss, output_attentions=output_attentions, return_dict=True)
        loss, loss_dict, auxiliary_predictions = (None, None, None)
        class_queries_logits = outputs.transformer_decoder_class_predictions
        masks_queries_logits = outputs.transformer_decoder_mask_predictions
        contrastive_queries_logits = outputs.transformer_decoder_contrastive_queries
        auxiliary_predictions = outputs.transformer_decoder_auxiliary_predictions
        text_queries = outputs.text_queries
        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(masks_queries_logits=masks_queries_logits, class_queries_logits=class_queries_logits, contrastive_queries_logits=contrastive_queries_logits, mask_labels=mask_labels, class_labels=class_labels, text_queries=text_queries, auxiliary_predictions=auxiliary_predictions, calculate_contrastive_loss=self.config.contrastive_temperature is not None)
            loss = self.get_loss(loss_dict)
        output_auxiliary_logits = self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        if not output_auxiliary_logits:
            auxiliary_predictions = None
        output = OneFormerForUniversalSegmentationOutput(class_queries_logits=class_queries_logits, masks_queries_logits=masks_queries_logits, auxiliary_predictions=auxiliary_predictions, loss=loss, **outputs)
        if not return_dict:
            output = tuple((v for v in output.values()))
            if loss is not None:
                output = loss + output
        return output