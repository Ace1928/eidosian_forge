import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@add_start_docstrings('The bare SegGpt Model transformer outputting raw hidden-states without any specific head on top.', SEGGPT_START_DOCSTRING)
class SegGptModel(SegGptPreTrainedModel):

    def __init__(self, config: SegGptConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = SegGptEmbeddings(config)
        self.encoder = SegGptEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> SegGptPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SEGGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SegGptEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.Tensor, prompt_pixel_values: torch.Tensor, prompt_masks: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor]=None, feature_ensemble: Optional[bool]=None, embedding_type: Optional[str]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SegGptEncoderOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import SegGptImageProcessor, SegGptModel
        >>> from PIL import Image
        >>> import requests

        >>> image_input_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
        >>> image_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1.jpg"
        >>> mask_prompt_url = "https://raw.githubusercontent.com/baaivision/Painter/main/SegGPT/SegGPT_inference/examples/hmbb_1_target.png"

        >>> image_input = Image.open(requests.get(image_input_url, stream=True).raw)
        >>> image_prompt = Image.open(requests.get(image_prompt_url, stream=True).raw)
        >>> mask_prompt = Image.open(requests.get(mask_prompt_url, stream=True).raw).convert("L")

        >>> checkpoint = "BAAI/seggpt-vit-large"
        >>> model = SegGptModel.from_pretrained(checkpoint)
        >>> image_processor = SegGptImageProcessor.from_pretrained(checkpoint)

        >>> inputs = image_processor(images=image_input, prompt_images=image_prompt, prompt_masks=mask_prompt, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> list(outputs.last_hidden_state.shape)
        [1, 56, 28, 1024]
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        feature_ensemble = feature_ensemble if feature_ensemble is not None else False
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        pixel_values = pixel_values.to(expected_dtype)
        prompt_pixel_values = prompt_pixel_values.to(expected_dtype)
        pixel_values = torch.cat((prompt_pixel_values, pixel_values), dim=2)
        prompt_pixel_values = torch.cat((prompt_masks, prompt_masks), dim=2)
        if bool_masked_pos is None:
            num_patches = self.embeddings.patch_embeddings.num_patches
            bool_masked_pos = torch.zeros(num_patches, dtype=torch.bool).to(pixel_values.device)
            bool_masked_pos[num_patches // 2:] = 1
            bool_masked_pos = bool_masked_pos.unsqueeze(0)
        embedding_output = self.embeddings(pixel_values, prompt_pixel_values, embedding_type=embedding_type, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(embedding_output, feature_ensemble=feature_ensemble, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        return encoder_outputs