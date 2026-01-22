import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_deit import DeiTConfig
@add_start_docstrings('\n    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of\n    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.\n\n    .. warning::\n\n           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet\n           supported.\n    ', DEIT_START_DOCSTRING)
class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deit = DeiTModel(config, add_pooling_layer=False)
        self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.distillation_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=DeiTForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, DeiTForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.deit(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
        logits = (cls_logits + distillation_logits) / 2
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output
        return DeiTForImageClassificationWithTeacherOutput(logits=logits, cls_logits=cls_logits, distillation_logits=distillation_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)