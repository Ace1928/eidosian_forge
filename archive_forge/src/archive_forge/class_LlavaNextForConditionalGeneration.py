from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...image_processing_utils import select_best_resolution
from ...modeling_outputs import ModelOutput
from ...utils import (
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava_next import LlavaNextConfig
@add_start_docstrings('The LLAVA-NeXT model which consists of a vision backbone and a language model.', LLAVA_NEXT_START_DOCSTRING)
class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel):

    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=self.dtype))
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config, attn_implementation=config._attn_implementation)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        max_embed_dim = num_special_image_tokens.max() * (num_image_patches - 1) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        new_token_positions = torch.cumsum(special_image_token_mask * (num_image_patches - 1) + 1, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
        final_embedding = torch.zeros(batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
        if labels is not None:
            final_labels = torch.full((batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device)
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (batch_indices.to(target_device), non_image_indices.to(target_device), text_to_overwrite.to(target_device))
        attention_mask = attention_mask.to(target_device)
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(f'The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation.')
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(final_attention_mask == 0, 1)
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        final_embedding[batch_indices, indices_to_mask] = 0
        if labels is None:
            final_labels = None
        return (final_embedding, final_attention_mask, final_labels, position_ids)

    @add_start_docstrings_to_model_forward(LLAVA_NEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaNextCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor=None, pixel_values: torch.FloatTensor=None, image_sizes: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, vision_feature_layer: Optional[int]=None, vision_feature_select_strategy: Optional[str]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \\nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = vision_feature_select_strategy if vision_feature_select_strategy is not None else self.config.vision_feature_select_strategy
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None and input_ids.shape[1] != 1:
                batch_size, num_patches, num_channels, height, width = pixel_values.shape
                reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
                image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]
                if vision_feature_select_strategy == 'default':
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == 'full':
                    selected_image_feature = selected_image_feature
                image_features = self.multi_modal_projector(selected_image_feature)
                split_sizes = [image.shape[0] for image in pixel_values]
                image_features = torch.split(image_features, split_sizes, dim=0)
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        if height * width != base_image_feature.shape[0]:
                            raise ValueError('The number of patches is not consistent with the image size.')
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.config.vision_config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                image_features = torch.stack(new_image_features, dim=0)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, labels)
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            elif past_key_values is not None and pixel_values is not None and (input_ids.shape[1] == 1):
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]
                extended_attention_mask = torch.ones((attention_mask.shape[0], past_length), dtype=attention_mask.dtype, device=attention_mask.device)
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        outputs = self.language_model(attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        logits = outputs[0]
        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return LlavaNextCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, image_sizes=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1:]
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]):]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'attention_mask': attention_mask, 'pixel_values': pixel_values, 'image_sizes': image_sizes})
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)