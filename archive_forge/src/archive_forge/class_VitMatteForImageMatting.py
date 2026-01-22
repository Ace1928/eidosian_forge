from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
@add_start_docstrings('ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.', VITMATTE_START_DOCSTRING)
class VitMatteForImageMatting(VitMattePreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.backbone = load_backbone(config)
        self.decoder = VitMatteDetailCaptureModule(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.Tensor]=None, return_dict: Optional[bool]=None):
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth image matting for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        ... )
        >>> trimap = Image.open(filepath).convert("L")

        >>> # prepare image + trimap for the model
        >>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

        >>> with torch.no_grad():
        ...     alphas = model(**inputs).alphas
        >>> print(alphas.shape)
        torch.Size([1, 1, 640, 960])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        features = outputs.feature_maps[-1]
        alphas = self.decoder(features, pixel_values)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not yet supported')
        if not return_dict:
            output = (alphas,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return ImageMattingOutput(loss=loss, alphas=alphas, hidden_states=outputs.hidden_states, attentions=outputs.attentions)