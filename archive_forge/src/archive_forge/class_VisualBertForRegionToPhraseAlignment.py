import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_visual_bert import VisualBertConfig
@add_start_docstrings('\n    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment\n    e.g. for Flickr30 Entities task.\n    ', VISUAL_BERT_START_DOCSTRING)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.bias']

    def __init__(self, config):
        super().__init__(config)
        self.visual_bert = VisualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = VisualBertPreTrainingHeads(config)
        self.attention = VisualBertRegionToPhraseAttention(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, visual_embeds: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.LongTensor]=None, visual_token_type_ids: Optional[torch.LongTensor]=None, image_text_alignment: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, region_to_phrase_position: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        region_to_phrase_position (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            The positions depicting the position of the image embedding corresponding to the textual tokens.

        labels (`torch.LongTensor` of shape `(batch_size, total_sequence_length, visual_sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and the
            outputs from the attention layer.

        Returns:

        Example:

        ```python
        # Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
        from transformers import AutoTokenizer, VisualBertForRegionToPhraseAlignment
        import torch

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = VisualBertForRegionToPhraseAlignment.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        text = "Who is eating the apple?"
        inputs = tokenizer(text, return_tensors="pt")
        visual_embeds = get_visual_embeddings(image).unsqueeze(0)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]))

        inputs.update(
            {
                "region_to_phrase_position": region_to_phrase_position,
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        labels = torch.ones(
            (1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2], visual_embeds.shape[-2])
        )  # Batch size 1

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        scores = outputs.logits
        ```"""
        if region_to_phrase_position is None:
            raise ValueError('`region_to_phrase_position` should not be None when using Flickr Model.')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.visual_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask, visual_token_type_ids=visual_token_type_ids, image_text_alignment=image_text_alignment, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        region_to_phrase_position_mask = (region_to_phrase_position != -1).long()
        region_to_phrase_position = region_to_phrase_position * region_to_phrase_position_mask
        expanded_region_to_phrase_positions = region_to_phrase_position.unsqueeze(2).expand(region_to_phrase_position.size(0), region_to_phrase_position.size(1), sequence_output.size(2))
        selected_positions = sequence_output.gather(1, expanded_region_to_phrase_positions)
        visual_features = sequence_output[:, attention_mask.size(1):]
        if visual_features.size(1) != visual_attention_mask.size(1):
            raise ValueError(f'Visual features length :{visual_features.size(1)} should be the same as visual attention mask length: {visual_attention_mask.size(1)}.')
        logits = self.attention(selected_positions, visual_features, visual_attention_mask)
        loss = None
        if labels is not None:
            loss_fct = KLDivLoss(reduction='batchmean')
            log_softmax = LogSoftmax(dim=-1)
            scores = log_softmax(logits)
            labels = labels.contiguous()
            loss = loss_fct(scores, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)