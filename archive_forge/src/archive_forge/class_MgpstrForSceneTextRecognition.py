import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
@add_start_docstrings('\n    MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top\n    of the transformer encoder output) for scene text recognition (STR) .\n    ', MGP_STR_START_DOCSTRING)
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    config_class = MgpstrConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: MgpstrConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mgp_str = MgpstrModel(config)
        self.char_a3_module = MgpstrA3Module(config)
        self.bpe_a3_module = MgpstrA3Module(config)
        self.wp_a3_module = MgpstrA3Module(config)
        self.char_head = nn.Linear(config.hidden_size, config.num_character_labels)
        self.bpe_head = nn.Linear(config.hidden_size, config.num_bpe_labels)
        self.wp_head = nn.Linear(config.hidden_size, config.num_wordpiece_labels)

    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MgpstrModelOutput, config_class=MgpstrConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool]=None, output_a3_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], MgpstrModelOutput]:
        """
        output_a3_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
            for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     MgpstrProcessor,
        ...     MgpstrForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '["ticket"]'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mgp_outputs = self.mgp_str(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = mgp_outputs[0]
        char_a3_out, char_attention = self.char_a3_module(sequence_output)
        bpe_a3_out, bpe_attention = self.bpe_a3_module(sequence_output)
        wp_a3_out, wp_attention = self.wp_a3_module(sequence_output)
        char_logits = self.char_head(char_a3_out)
        bpe_logits = self.bpe_head(bpe_a3_out)
        wp_logits = self.wp_head(wp_a3_out)
        all_a3_attentions = (char_attention, bpe_attention, wp_attention) if output_a3_attentions else None
        all_logits = (char_logits, bpe_logits, wp_logits)
        if not return_dict:
            outputs = (all_logits, all_a3_attentions) + mgp_outputs[1:]
            return tuple((output for output in outputs if output is not None))
        return MgpstrModelOutput(logits=all_logits, hidden_states=mgp_outputs.hidden_states, attentions=mgp_outputs.attentions, a3_attentions=all_a3_attentions)