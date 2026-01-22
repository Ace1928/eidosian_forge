import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
@add_start_docstrings('The bare PatchTST Model outputting raw hidden-states without any specific head.', PATCHTST_START_DOCSTRING)
class PatchTSTModel(PatchTSTPreTrainedModel):

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        self.do_mask_input = config.do_mask_input
        num_patches = self.patchifier.num_patches
        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            self.masking = nn.Identity()
        self.encoder = PatchTSTEncoder(config, num_patches=num_patches)
        self.post_init()

    def forward(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, PatchTSTModelOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTModelOutput` or tuple of `torch.Tensor` (if `return_dict`=False or `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTModel

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = PatchTSTModel.from_pretrained("namctin/patchtst_etth1_pretrain")

        >>> # during training, one provides both past and future values
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     future_values=batch["future_values"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        patched_values = self.patchifier(scaled_past_values)
        if self.do_mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = (self.masking(patched_values), None)
        encoder_output = self.encoder(patch_input=masked_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        if not return_dict:
            outputs = (encoder_output.last_hidden_state, encoder_output.hidden_states, encoder_output.attentions)
            outputs = outputs + (mask, loc, scale, patched_values)
            return tuple((v for v in outputs if v is not None))
        return PatchTSTModelOutput(last_hidden_state=encoder_output.last_hidden_state, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions, mask=mask, loc=loc, scale=scale, patch_input=patched_values)