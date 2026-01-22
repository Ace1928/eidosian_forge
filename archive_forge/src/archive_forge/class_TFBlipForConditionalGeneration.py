from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
@add_start_docstrings('\n    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass\n    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,\n    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption\n    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.\n    ', BLIP_START_DOCSTRING)
class TFBlipForConditionalGeneration(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']
    main_input_name = 'pixel_values'

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name='text_decoder')
        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipForConditionalGenerationModelOutput, config_class=BlipConfig)
    def call(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipForConditionalGenerationModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        outputs = self.text_decoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, labels=labels, return_dict=False, training=training)
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        if labels is not None:
            loss = outputs[0]
            logits = outputs[1]
        else:
            loss = None
            logits = outputs[0]
        if loss is not None and loss.shape.rank == 0:
            loss = tf.reshape(loss, (1,))
        return TFBlipForConditionalGenerationModelOutput(loss=loss, logits=logits, image_embeds=image_embeds, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions)

    def generate(self, pixel_values: tf.Tensor, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, **generate_kwargs) -> tf.Tensor:
        """
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForConditionalGeneration

        >>> model = TFBlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```
        """
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)
        if isinstance(input_ids, list):
            input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        elif input_ids is None:
            input_ids = tf.convert_to_tensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]], dtype=tf.int32)
            input_ids = tf.tile(input_ids, (batch_size, 1))
        input_ids = tf.concat([tf.ones((batch_size, 1), dtype=tf.int32) * self.config.text_config.bos_token_id, input_ids[:, 1:]], axis=1)
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None
        outputs = self.text_decoder.generate(input_ids=input_ids[:, :-1], eos_token_id=self.config.text_config.sep_token_id, pad_token_id=self.config.text_config.pad_token_id, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, **generate_kwargs)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'vision_model', None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, 'text_decoder', None) is not None:
            with tf.name_scope(self.text_decoder.name):
                self.text_decoder.build(None)