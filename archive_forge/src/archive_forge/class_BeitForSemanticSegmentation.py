import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig
@add_start_docstrings('\n    Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.\n    ', BEIT_START_DOCSTRING)
class BeitForSemanticSegmentation(BeitPreTrainedModel):

    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)
        if len(self.config.out_indices) != 4:
            raise ValueError('BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of a base-sized architecture.')
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2), nn.BatchNorm2d(config.hidden_size), nn.GELU(), nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(auxiliary_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss
        return loss

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, SemanticSegmenterOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, BeitForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        >>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.beit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features]
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        logits = self.decode_head(features)
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)