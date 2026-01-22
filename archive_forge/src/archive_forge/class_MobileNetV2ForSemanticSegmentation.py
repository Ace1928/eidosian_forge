from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
@add_start_docstrings('\n    MobileNetV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.\n    ', MOBILENET_V2_START_DOCSTRING)
class MobileNetV2ForSemanticSegmentation(MobileNetV2PreTrainedModel):

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mobilenet_v2 = MobileNetV2Model(config, add_pooling_layer=False)
        self.segmentation_head = MobileNetV2DeepLabV3Plus(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, SemanticSegmenterOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MobileNetV2ForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        >>> model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=True, return_dict=return_dict)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        logits = self.segmentation_head(encoder_hidden_states[-1])
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SemanticSegmenterOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=None)