import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_bros import BrosConfig
@add_start_docstrings('\n    Bros Model with a token classification head on top (a entity_linker layer on top of the hidden-states output) e.g.\n    for Entity-Linking. The entity_linker is used to predict intra-entity links (one entity to another entity).\n    ', BROS_START_DOCSTRING)
class BrosSpadeELForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.bros = BrosModel(config)
        config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.hidden_dropout_prob
        self.entity_linker = BrosRelationExtractor(config)
        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, bbox: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, bbox_first_token_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bros(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()
        logits = self.entity_linker(last_hidden_states, last_hidden_states).squeeze(0)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            batch_size, max_seq_length = attention_mask.shape
            device = attention_mask.device
            self_token_mask = torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
            mask = bbox_first_token_mask.view(-1)
            bbox_first_token_mask = torch.cat([~bbox_first_token_mask, torch.zeros([batch_size, 1], dtype=torch.bool).to(device)], axis=1)
            logits = logits.masked_fill(bbox_first_token_mask[:, None, :], torch.finfo(logits.dtype).min)
            logits = logits.masked_fill(self_token_mask[None, :, :], torch.finfo(logits.dtype).min)
            loss = loss_fct(logits.view(-1, max_seq_length + 1)[mask], labels.view(-1)[mask])
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)