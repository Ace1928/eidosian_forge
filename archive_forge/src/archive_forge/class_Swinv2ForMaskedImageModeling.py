import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_swinv2 import Swinv2Config
@add_start_docstrings('Swinv2 Model with a decoder on top for masked image modeling, as proposed in\n[SimMIM](https://arxiv.org/abs/2111.09886).\n\n    <Tip>\n\n    Note that we provide a script to pre-train this model on custom data in our [examples\n    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).\n\n    </Tip>\n    ', SWINV2_START_DOCSTRING)
class Swinv2ForMaskedImageModeling(Swinv2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.swinv2 = Swinv2Model(config, add_pooling_layer=False, use_mask_token=True)
        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=num_features, out_channels=config.encoder_stride ** 2 * config.num_channels, kernel_size=1), nn.PixelShuffle(config.encoder_stride))
        self.post_init()

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Swinv2MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, bool_masked_pos: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Swinv2MaskedImageModelingOutput]:
        """
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, Swinv2ForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        >>> model = Swinv2ForMaskedImageModeling.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 256, 256]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swinv2(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)
        reconstructed_pixel_values = self.decoder(sequence_output)
        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = bool_masked_pos.repeat_interleave(self.config.patch_size, 1).repeat_interleave(self.config.patch_size, 2).unsqueeze(1).contiguous()
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction='none')
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-05) / self.config.num_channels
        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[2:]
            return (masked_im_loss,) + output if masked_im_loss is not None else output
        return Swinv2MaskedImageModelingOutput(loss=masked_im_loss, reconstruction=reconstructed_pixel_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, reshaped_hidden_states=outputs.reshaped_hidden_states)