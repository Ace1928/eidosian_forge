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
@add_start_docstrings('The PatchTST for prediction model.', PATCHTST_START_DOCSTRING)
class PatchTSTForPrediction(PatchTSTPreTrainedModel):

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        if config.do_mask_input:
            logger.warning('Setting `do_mask_input` parameter to False.')
            config.do_mask_input = False
        self.model = PatchTSTModel(config)
        if config.loss == 'mse':
            self.distribution_output = None
        elif config.distribution_output == 'student_t':
            self.distribution_output = StudentTOutput(dim=config.prediction_length)
        elif config.distribution_output == 'normal':
            self.distribution_output = NormalOutput(dim=config.prediction_length)
        elif config.distribution_output == 'negative_binomial':
            self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
        else:
            raise ValueError(f'Unknown distribution output {config.distribution_output}')
        self.head = PatchTSTPredictionHead(config, self.model.patchifier.num_patches, distribution_output=self.distribution_output)
        self.post_init()

    def forward(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None, future_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, PatchTSTForPredictionOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.Tensor` of shape `(bs, forecast_len, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPredictionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTConfig, PatchTSTForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> # Prediction task with 7 input channels and prediction length is 96
        >>> model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")

        >>> # during training, one provides both past and future values
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     future_values=batch["future_values"],
        ... )

        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # during inference, one only provides past values, the model outputs future values
        >>> outputs = model(past_values=batch["past_values"])
        >>> prediction_outputs = outputs.prediction_outputs
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_output = self.model(past_values=past_values, past_observed_mask=past_observed_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=True)
        y_hat = self.head(model_output.last_hidden_state)
        loss_val = None
        if self.distribution_output:
            y_hat_out = y_hat
        else:
            y_hat_out = y_hat * model_output.scale + model_output.loc
        if future_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat, loc=model_output.loc, scale=model_output.scale)
                loss_val = nll(distribution, future_values)
                loss_val = weighted_average(loss_val)
            else:
                loss = nn.MSELoss(reduction='mean')
                loss_val = loss(y_hat_out, future_values)
        loc = model_output.loc
        scale = model_output.scale
        if not return_dict:
            outputs = (y_hat_out,) + model_output[1:-1]
            outputs = (loss_val,) + outputs if loss_val is not None else outputs
            return outputs
        return PatchTSTForPredictionOutput(loss=loss_val, prediction_outputs=y_hat_out, hidden_states=model_output.hidden_states, attentions=model_output.attentions, loc=loc, scale=scale)

    def generate(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor]=None) -> SamplePatchTSTOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, prediction_length, 1)` or `(batch_size, number of samples, prediction_length, num_input_channels)`
            for multivariate predictions.
        """
        num_parallel_samples = self.config.num_parallel_samples
        outputs = self(past_values=past_values, future_values=None, past_observed_mask=past_observed_mask, output_hidden_states=False)
        if self.distribution_output:
            distribution = self.distribution_output.distribution(outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale)
            samples = [distribution.sample() for _ in range(num_parallel_samples)]
            samples = torch.stack(samples, dim=1)
        else:
            samples = outputs.prediction_outputs.unsqueeze(1)
        return SamplePatchTSTOutput(sequences=samples)