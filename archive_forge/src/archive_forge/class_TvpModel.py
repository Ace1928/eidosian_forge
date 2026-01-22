import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
@add_start_docstrings('The bare Tvp Model transformer outputting BaseModelOutputWithPooling object without any specific head on top.', TVP_START_DOCSTRING)
class TvpModel(TvpPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = TvpVisionModel(config)
        self.embeddings = TvpTextInputEmbeddings(config)
        self.visual_embeddings = TvpVisualInputEmbedding(config)
        self.encoder = TvpEncoder(config)
        self.pooler = TvpPooler(config)
        self.text_prompt = nn.Parameter(torch.randn([1, 10, config.hidden_size]))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.visual_prompter_type not in TVP_PROMPTER_CLASSES_MAPPING:
            raise ValueError('`visual_prompter_type` must be in (framedownpad, framepad)')
        self.visual_prompter = TVP_PROMPTER_CLASSES_MAPPING[config.visual_prompter_type](config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=TvpConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoConfig, AutoTokenizer, TvpModel

        >>> model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = torch.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
        >>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = self.vision_model(self.visual_prompter(pixel_values))
        text_embedding_output = self.embeddings(input_ids=input_ids)
        visual_embedding_output = self.visual_embeddings(pixel_values)
        if attention_mask is not None:
            visual_attention_mask = attention_mask.new_ones(visual_embedding_output.shape[:2])
            pt_mask = torch.ones(attention_mask.shape[0], 10).to(device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pt_mask, attention_mask, visual_attention_mask], dim=-1)
            attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size()).to(input_ids.device)
        text_prompt = self.text_prompt.expand(text_embedding_output.shape[0], -1, -1)
        embedding_output = torch.cat([text_prompt, text_embedding_output, visual_embedding_output], dim=1)
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, head_mask=self.get_head_mask(head_mask, self.config.num_hidden_layers), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        last_hidden_state = self.dropout(last_hidden_state)
        pooled_output = self.dropout(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)