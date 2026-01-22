import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
@add_start_docstrings('\n    LayoutLMv2 Model with a sequence classification head on top (a linear layer on top of the concatenation of the\n    final hidden state of the [CLS] token, average-pooled initial visual embeddings and average-pooled final visual\n    embeddings, e.g. for document image classification tasks such as the\n    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.\n    ', LAYOUTLMV2_START_DOCSTRING)
class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, bbox: Optional[torch.LongTensor]=None, image: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SequenceClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, LayoutLMv2ForSequenceClassification, set_seed
        >>> from PIL import Image
        >>> import torch
        >>> from datasets import load_dataset

        >>> set_seed(88)

        >>> dataset = load_dataset("rvl_cdip", split="train", streaming=True)
        >>> data = next(iter(dataset))
        >>> image = data["image"].convert("RGB")

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2ForSequenceClassification.from_pretrained(
        ...     "microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes
        ... )

        >>> encoding = processor(image, return_tensors="pt")
        >>> sequence_label = torch.tensor([data["label"]])

        >>> outputs = model(**encoding, labels=sequence_label)

        >>> loss, logits = outputs.loss, outputs.logits
        >>> predicted_idx = logits.argmax(dim=-1).item()
        >>> predicted_answer = dataset.info.features["label"].names[4]
        >>> predicted_idx, predicted_answer
        (4, 'advertisement')
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)
        visual_bbox = self.layoutlmv2._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(input_shape[0], 1)
        initial_image_embeddings = self.layoutlmv2._calc_img_embeddings(image=image, bbox=visual_bbox, position_ids=visual_position_ids)
        outputs = self.layoutlmv2(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        sequence_output, final_image_embeddings = (outputs[0][:, :seq_length], outputs[0][:, seq_length:])
        cls_final_output = sequence_output[:, 0, :]
        pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
        sequence_output = torch.cat([cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings], dim=1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)