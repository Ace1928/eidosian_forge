import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
@add_start_docstrings('\n    SwiftFormer Model transformer with an image classification head on top (e.g. for ImageNet).\n    ', SWIFTFORMER_START_DOCSTRING)
class SwiftFormerForImageClassification(SwiftFormerPreTrainedModel):

    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__(config)
        embed_dims = config.embed_dims
        self.num_labels = config.num_labels
        self.swiftformer = SwiftFormerModel(config)
        self.norm = nn.BatchNorm2d(embed_dims[-1], eps=config.batch_norm_eps)
        self.head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()
        self.dist_head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIFTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swiftformer(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]
        sequence_output = self.norm(sequence_output)
        sequence_output = sequence_output.flatten(2).mean(-1)
        cls_out = self.head(sequence_output)
        distillation_out = self.dist_head(sequence_output)
        logits = (cls_out + distillation_out) / 2
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)