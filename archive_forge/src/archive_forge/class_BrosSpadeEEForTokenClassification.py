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
@add_start_docstrings('\n    Bros Model with a token classification head on top (initial_token_layers and subsequent_token_layer on top of the\n    hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. The initial_token_classifier is used to\n    predict the first token of each entity, and the subsequent_token_classifier is used to predict the subsequent\n    tokens within an entity. Compared to BrosForTokenClassification, this model is more robust to serialization errors\n    since it predicts next token from one token.\n    ', BROS_START_DOCSTRING)
class BrosSpadeEEForTokenClassification(BrosPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler']

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.bros = BrosModel(config)
        classifier_dropout = config.classifier_dropout if hasattr(config, 'classifier_dropout') else config.hidden_dropout_prob
        self.initial_token_classifier = nn.Sequential(nn.Dropout(classifier_dropout), nn.Linear(config.hidden_size, config.hidden_size), nn.Dropout(classifier_dropout), nn.Linear(config.hidden_size, config.num_labels))
        self.subsequent_token_classifier = BrosRelationExtractor(config)
        self.init_weights()

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BrosSpadeOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, bbox: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, bbox_first_token_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, initial_token_labels: Optional[torch.Tensor]=None, subsequent_token_labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], BrosSpadeOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeEEForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bros(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()
        initial_token_logits = self.initial_token_classifier(last_hidden_states).transpose(0, 1).contiguous()
        subsequent_token_logits = self.subsequent_token_classifier(last_hidden_states, last_hidden_states).squeeze(0)
        inv_attention_mask = 1 - attention_mask
        batch_size, max_seq_length = inv_attention_mask.shape
        device = inv_attention_mask.device
        invalid_token_mask = torch.cat([inv_attention_mask, torch.zeros([batch_size, 1]).to(device)], axis=1).bool()
        subsequent_token_logits = subsequent_token_logits.masked_fill(invalid_token_mask[:, None, :], torch.finfo(subsequent_token_logits.dtype).min)
        self_token_mask = torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()
        subsequent_token_logits = subsequent_token_logits.masked_fill(self_token_mask[None, :, :], torch.finfo(subsequent_token_logits.dtype).min)
        subsequent_token_mask = attention_mask.view(-1).bool()
        loss = None
        if initial_token_labels is not None and subsequent_token_labels is not None:
            loss_fct = CrossEntropyLoss()
            initial_token_labels = initial_token_labels.view(-1)
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                initial_token_loss = loss_fct(initial_token_logits.view(-1, self.num_labels)[bbox_first_token_mask], initial_token_labels[bbox_first_token_mask])
            else:
                initial_token_loss = loss_fct(initial_token_logits.view(-1, self.num_labels), initial_token_labels)
            subsequent_token_labels = subsequent_token_labels.view(-1)
            subsequent_token_loss = loss_fct(subsequent_token_logits.view(-1, max_seq_length + 1)[subsequent_token_mask], subsequent_token_labels[subsequent_token_mask])
            loss = initial_token_loss + subsequent_token_loss
        if not return_dict:
            output = (initial_token_logits, subsequent_token_logits) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return BrosSpadeOutput(loss=loss, initial_token_logits=initial_token_logits, subsequent_token_logits=subsequent_token_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)