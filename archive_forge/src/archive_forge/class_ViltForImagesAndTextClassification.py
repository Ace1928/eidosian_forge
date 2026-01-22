import collections.abc
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
from ...pytorch_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig
@add_start_docstrings('\n    Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.\n    ', VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING)
class ViltForImagesAndTextClassification(ViltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)
        num_images = config.num_images
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images), nn.LayerNorm(config.hidden_size * num_images), nn.GELU(), nn.Linear(config.hidden_size * num_images, config.num_labels))
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViltForImagesAndTextClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[ViltForImagesAndTextClassificationOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Binary classification labels.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForImagesAndTextClassification
        >>> import requests
        >>> from PIL import Image

        >>> image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
        >>> image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg", stream=True).raw)
        >>> text = "The left image contains twice the number of dogs as the right image."

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
        >>> model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

        >>> # prepare inputs
        >>> encoding = processor([image1, image2], text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
        >>> logits = outputs.logits
        >>> idx = logits.argmax(-1).item()
        >>> print("Predicted answer:", model.config.id2label[idx])
        Predicted answer: True
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is not None and pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if image_embeds is not None and image_embeds.ndim == 3:
            image_embeds = image_embeds.unsqueeze(1)
        num_images = pixel_values.shape[1] if pixel_values is not None else None
        if num_images is None:
            num_images = image_embeds.shape[1] if image_embeds is not None else None
        if num_images != self.config.num_images:
            raise ValueError('Make sure to match the number of images in the model with the number of images in the input.')
        pooler_outputs = []
        hidden_states = [] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for i in range(num_images):
            outputs = self.vilt(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values[:, i, :, :, :] if pixel_values is not None else None, pixel_mask=pixel_mask[:, i, :, :] if pixel_mask is not None else None, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds[:, i, :, :] if image_embeds is not None else None, image_token_type_idx=i + 1, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            pooler_output = outputs.pooler_output if return_dict else outputs[1]
            pooler_outputs.append(pooler_output)
            if output_hidden_states:
                hidden_states.append(outputs.hidden_states)
            if output_attentions:
                attentions.append(outputs.attentions)
        pooled_output = torch.cat(pooler_outputs, dim=-1)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits, hidden_states, attentions)
            return (loss,) + output if loss is not None else output
        return ViltForImagesAndTextClassificationOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)