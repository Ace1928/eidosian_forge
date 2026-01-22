import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import normalize
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_blip_text import BlipTextLMHeadModel, BlipTextModel
@add_start_docstrings('\n    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of\n    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to\n    the image.\n    ', BLIP_START_DOCSTRING)
class BlipForImageTextRetrieval(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)
        self.decoder_pad_token_id = config.text_config.pad_token_id if not hasattr(config, 'decoder_pad_token_id') else config.decoder_pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id if not hasattr(config, 'decoder_start_token_id') else config.decoder_start_token_id
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, use_itm_head: Optional[bool]=True, attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BlipTextVisionModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
        if use_itm_head:
            question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=return_dict)
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
            output = self.itm_head(question_embeds[:, 0, :])
        else:
            question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
            output = image_feat @ text_feat.t()
        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple((output for output in outputs if output is not None))
        return BlipImageTextMatchingModelOutput(itm_score=output, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions, question_embeds=question_embeds)