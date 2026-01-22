import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_yolos import YolosConfig
@add_start_docstrings('\n    YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.\n    ', YOLOS_START_DOCSTRING)
class YolosForObjectDetection(YolosPreTrainedModel):

    def __init__(self, config: YolosConfig):
        super().__init__(config)
        self.vit = YolosModel(config, add_pooling_layer=False)
        self.class_labels_classifier = YolosMLPPredictionHead(input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3)
        self.bbox_predictor = YolosMLPPredictionHead(input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3)
        self.post_init()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=YolosObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[List[Dict]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, YolosObjectDetectionOutput]:
        """
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding
            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,
            4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.994 at location [46.96, 72.61, 181.02, 119.73]
        Detected remote with confidence 0.975 at location [340.66, 79.19, 372.59, 192.65]
        Detected cat with confidence 0.984 at location [12.27, 54.25, 319.42, 470.99]
        Detected remote with confidence 0.922 at location [41.66, 71.96, 178.7, 120.33]
        Detected cat with confidence 0.914 at location [342.34, 21.48, 638.64, 372.46]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, -self.config.num_detection_tokens:, :]
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        loss, loss_dict, auxiliary_outputs = (None, None, None)
        if labels is not None:
            matcher = YolosHungarianMatcher(class_cost=self.config.class_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost)
            losses = ['labels', 'boxes', 'cardinality']
            criterion = YolosLoss(matcher=matcher, num_classes=self.config.num_labels, eos_coef=self.config.eos_coefficient, losses=losses)
            criterion.to(self.device)
            outputs_loss = {}
            outputs_loss['logits'] = logits
            outputs_loss['pred_boxes'] = pred_boxes
            if self.config.auxiliary_loss:
                intermediate = outputs.intermediate_hidden_states if return_dict else outputs[4]
                outputs_class = self.class_labels_classifier(intermediate)
                outputs_coord = self.bbox_predictor(intermediate).sigmoid()
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord)
                outputs_loss['auxiliary_outputs'] = auxiliary_outputs
            loss_dict = criterion(outputs_loss, labels)
            weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coefficient}
            weight_dict['loss_giou'] = self.config.giou_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum((loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict))
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return (loss, loss_dict) + output if loss is not None else output
        return YolosObjectDetectionOutput(loss=loss, loss_dict=loss_dict, logits=logits, pred_boxes=pred_boxes, auxiliary_outputs=auxiliary_outputs, last_hidden_state=outputs.last_hidden_state, hidden_states=outputs.hidden_states, attentions=outputs.attentions)