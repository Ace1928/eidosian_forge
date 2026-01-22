import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
@add_start_docstrings('\n    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.\n    ', CLIPSEG_START_DOCSTRING)
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)
        self.config = config
        self.clip = CLIPSegModel(config)
        self.extract_layers = config.extract_layers
        self.decoder = CLIPSegDecoder(config)
        self.post_init()

    def get_conditional_embeddings(self, batch_size: int=None, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, conditional_pixel_values: Optional[torch.Tensor]=None):
        if input_ids is not None:
            if len(input_ids) != batch_size:
                raise ValueError('Make sure to pass as many prompt texts as there are query images')
            with torch.no_grad():
                conditional_embeddings = self.clip.get_text_features(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        elif conditional_pixel_values is not None:
            if len(conditional_pixel_values) != batch_size:
                raise ValueError('Make sure to pass as many prompt images as there are query images')
            with torch.no_grad():
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            raise ValueError('Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`')
        return conditional_embeddings

    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegImageSegmentationOutput, config_class=CLIPSegTextConfig)
    def forward(self, input_ids: Optional[torch.FloatTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, conditional_pixel_values: Optional[torch.FloatTensor]=None, conditional_embeddings: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, CLIPSegOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        torch.Size([3, 352, 352])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
            pooled_output = self.clip.visual_projection(vision_outputs[1])
            hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
            activations = [hidden_states[i + 1] for i in self.extract_layers]
            if return_dict:
                vision_outputs = BaseModelOutputWithPooling(last_hidden_state=vision_outputs.last_hidden_state, pooler_output=vision_outputs.pooler_output, hidden_states=vision_outputs.hidden_states if output_hidden_states else None, attentions=vision_outputs.attentions)
            else:
                vision_outputs = vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(batch_size=pixel_values.shape[0], input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, conditional_pixel_values=conditional_pixel_values)
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError('Make sure to pass as many conditional embeddings as there are query images in the batch')
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError('Make sure that the feature dimension of the conditional embeddings matches `config.projection_dim`.')
        decoder_outputs = self.decoder(activations, conditional_embeddings, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs)
            return (loss,) + output if loss is not None else output
        return CLIPSegImageSegmentationOutput(loss=loss, logits=logits, conditional_embeddings=conditional_embeddings, pooled_output=pooled_output, vision_model_output=vision_outputs, decoder_output=decoder_outputs)