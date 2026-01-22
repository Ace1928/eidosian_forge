import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
@add_start_docstrings('\n    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and\n    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::\n           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet\n           supported.\n    ', LEVIT_START_DOCSTRING)
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = LevitModel(config)
        self.classifier = LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else torch.nn.Identity()
        self.classifier_distill = LevitClassificationLayer(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else torch.nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=LevitForImageClassificationWithTeacherOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: torch.FloatTensor=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LevitForImageClassificationWithTeacherOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)
        cls_logits, distill_logits = (self.classifier(sequence_output), self.classifier_distill(sequence_output))
        logits = (cls_logits + distill_logits) / 2
        if not return_dict:
            output = (logits, cls_logits, distill_logits) + outputs[2:]
            return output
        return LevitForImageClassificationWithTeacherOutput(logits=logits, cls_logits=cls_logits, distillation_logits=distill_logits, hidden_states=outputs.hidden_states)