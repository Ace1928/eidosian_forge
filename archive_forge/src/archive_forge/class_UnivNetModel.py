from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig
@add_start_docstrings('UnivNet GAN vocoder.', UNIVNET_START_DOCSTRING)
class UnivNetModel(PreTrainedModel):
    config_class = UnivNetConfig
    main_input_name = 'input_features'

    def __init__(self, config: UnivNetConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.leaky_relu_slope = config.leaky_relu_slope
        self.conv_pre = nn.Conv1d(config.model_in_channels, config.model_hidden_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        num_layers = len(config.resblock_stride_sizes)
        hop_length = 1
        hop_lengths = []
        for stride in config.resblock_stride_sizes:
            hop_length = hop_length * stride
            hop_lengths.append(hop_length)
        self.resblocks = nn.ModuleList([UnivNetLvcBlock(config, layer_id=i, lvc_hop_size=hop_lengths[i]) for i in range(num_layers)])
        self.conv_post = nn.Conv1d(config.model_hidden_channels, 1, 7, padding=3, padding_mode='reflect')
        self.post_init()

    @add_start_docstrings_to_model_forward(UNIVNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UnivNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_features: torch.FloatTensor, noise_sequence: Optional[torch.FloatTensor]=None, padding_mask: Optional[torch.FloatTensor]=None, generator: Optional[torch.Generator]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], UnivNetModelOutput]:
        """
        Returns:

        Example:

         ```python
         >>> from transformers import UnivNetFeatureExtractor, UnivNetModel
         >>> from datasets import load_dataset, Audio

         >>> model = UnivNetModel.from_pretrained("dg845/univnet-dev")
         >>> feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")

         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> # Resample the audio to the feature extractor's sampling rate.
         >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
         >>> inputs = feature_extractor(
         ...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
         ... )
         >>> audio = model(**inputs).waveforms
         >>> list(audio.shape)
         [1, 140288]
         ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        spectrogram_batched = input_features.dim() == 3
        if not spectrogram_batched:
            input_features = input_features.unsqueeze(0)
        spectrogram_batch_size, spectrogram_length, _ = input_features.shape
        if noise_sequence is not None:
            noise_sequence_batched = noise_sequence.dim() == 3
            if not noise_sequence_batched:
                noise_sequence = noise_sequence.unsqueeze(0)
        else:
            noise_sequence_shape = (spectrogram_batch_size, spectrogram_length, self.config.model_in_channels)
            noise_sequence = torch.randn(noise_sequence_shape, generator=generator, dtype=input_features.dtype, device=input_features.device)
        noise_sequence_batch_size = noise_sequence.shape[0]
        if spectrogram_batch_size > 1 and noise_sequence_batch_size == 1:
            noise_sequence = noise_sequence.repeat(spectrogram_batch_size, 1, 1)
        elif noise_sequence_batch_size > 1 and spectrogram_batch_size == 1:
            input_features = input_features.repeat(noise_sequence_batch_size, 1, 1)
        if noise_sequence_batch_size != spectrogram_batch_size:
            raise ValueError(f'The batch size of `noise_sequence` is {noise_sequence_batch_size} and the batch size of `input_features` is {spectrogram_batch_size}, but the two are expected to be equal.')
        if padding_mask is not None:
            if padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)
            padding_mask_batch_size = padding_mask.shape[0]
            if padding_mask_batch_size != spectrogram_batch_size:
                raise ValueError(f'The batch size of `padding_mask` is {padding_mask_batch_size} and the batch size of `input_features` is {spectrogram_batch_size}, but the two are expected to be equal.')
        hidden_states = noise_sequence.transpose(2, 1)
        input_features = input_features.transpose(2, 1)
        hidden_states = self.conv_pre(hidden_states)
        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states, input_features)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        waveform = hidden_states.squeeze(1)
        waveform_lengths = None
        if padding_mask is not None:
            waveform_lengths = torch.sum(padding_mask, dim=1)
        if not return_dict:
            outputs = (waveform, waveform_lengths)
            return outputs
        return UnivNetModelOutput(waveforms=waveform, waveform_lengths=waveform_lengths)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)