import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
@add_start_docstrings('\n    KOSMOS-2 Model for generating text and bounding boxes given an image. The model consists of a vision encoder and a\n    language model.\n    ', KOSMOS2_START_DOCSTRING)
class Kosmos2ForConditionalGeneration(Kosmos2PreTrainedModel):
    config_class = Kosmos2Config
    main_input_name = 'pixel_values'
    _tied_weights_keys = ['text_model.lm_head.weight']

    def __init__(self, config: Kosmos2Config):
        super().__init__(config)
        self.text_model = Kosmos2TextForCausalLM(config.text_config)
        self.vision_model = Kosmos2VisionModel(config.vision_config)
        self.image_to_text_projection = Kosmos2ImageToTextProjection(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.text_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_model.set_output_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(KOSMOS2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Kosmos2ForConditionalGenerationModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, input_ids: Optional[torch.Tensor]=None, image_embeds_position_mask: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, image_embeds: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Kosmos2ForConditionalGenerationModelOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

        >>> model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        >>> processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        >>> url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "<grounding> An image of"

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> generated_ids = model.generate(
        ...     pixel_values=inputs["pixel_values"],
        ...     input_ids=inputs["input_ids"],
        ...     attention_mask=inputs["attention_mask"],
        ...     image_embeds=None,
        ...     image_embeds_position_mask=inputs["image_embeds_position_mask"],
        ...     use_cache=True,
        ...     max_new_tokens=64,
        ... )
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
        >>> processed_text
        '<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.'

        >>> caption, entities = processor.post_process_generation(generated_text)
        >>> caption
        'An image of a snowman warming himself by a fire.'

        >>> entities
        [('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]), ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)])]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_model_output = None
        projection_attentions = None
        if image_embeds is None:
            if pixel_values is None:
                raise ValueError('You have to specify either `pixel_values` or `image_embeds`.')
            vision_model_output = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)
        lm_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, image_embeds=image_embeds, image_embeds_position_mask=image_embeds_position_mask, head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=inputs_embeds, position_ids=position_ids, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            outputs = lm_outputs + (image_embeds, projection_attentions, vision_model_output)
            return tuple((output for output in outputs if output is not None))
        return Kosmos2ForConditionalGenerationModelOutput(loss=lm_outputs.loss, logits=lm_outputs.logits, past_key_values=lm_outputs.past_key_values, hidden_states=lm_outputs.hidden_states, attentions=lm_outputs.attentions, image_embeds=image_embeds, projection_attentions=projection_attentions, vision_model_output=vision_model_output)

    def generate(self, pixel_values: Optional[torch.Tensor]=None, image_embeds_position_mask: Optional[torch.Tensor]=None, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, image_embeds: Optional[torch.Tensor]=None, **kwargs):
        inputs = kwargs.pop('inputs', None)
        if pixel_values is not None and inputs is not None:
            raise ValueError(f'`inputs`: {inputs} were passed alongside `pixel_values` which is not allowed.Make sure to either pass `inputs` or pixel_values=...')
        if pixel_values is None and inputs is not None:
            pixel_values = inputs
        if image_embeds is None:
            vision_model_output = self.vision_model(pixel_values)
            image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
            image_embeds = nn.functional.normalize(image_embeds, dim=-1)
            image_embeds, projection_attentions = self.image_to_text_projection(image_embeds)
        output = self.text_model.generate(input_ids=input_ids, attention_mask=attention_mask, image_embeds=image_embeds, image_embeds_position_mask=image_embeds_position_mask, **kwargs)
        return output