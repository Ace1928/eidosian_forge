import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
@add_start_docstrings('Lxmert Model with a visual-answering head on top for downstream QA tasks', LXMERT_START_DOCSTRING)
class LxmertForQuestionAnswering(LxmertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer
        self.lxmert = LxmertModel(config)
        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)
        self.post_init()
        self.loss = CrossEntropyLoss()

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``torch.nn.Linear``` module of the model without doing anything.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels
        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the linear layer that produces question answering logits

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states. `None`: A NoneType
            object if Lxmert does not have the visual answering head.
        """
        if hasattr(self, 'answer_head'):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer
        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer
        if getattr(cur_qa_logit_layer, 'bias', None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)
        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)
        self._init_weights(new_qa_logit_layer)
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, 'bias', None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]
        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=LxmertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, visual_feats: Optional[torch.FloatTensor]=None, visual_pos: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        lxmert_output = self.lxmert(input_ids=input_ids, visual_feats=visual_feats, visual_pos=visual_pos, token_type_ids=token_type_ids, attention_mask=attention_mask, visual_attention_mask=visual_attention_mask, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))
        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output
        return LxmertForQuestionAnsweringOutput(loss=loss, question_answering_score=answer_score, language_hidden_states=lxmert_output.language_hidden_states, vision_hidden_states=lxmert_output.vision_hidden_states, language_attentions=lxmert_output.language_attentions, vision_attentions=lxmert_output.vision_attentions, cross_encoder_attentions=lxmert_output.cross_encoder_attentions)