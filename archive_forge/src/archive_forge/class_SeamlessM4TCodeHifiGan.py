import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
@add_start_docstrings('Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).', HIFIGAN_START_DOCSTRING)
class SeamlessM4TCodeHifiGan(PreTrainedModel):
    config_class = SeamlessM4TConfig
    main_input_name = 'input_embeds'
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.t2u_pad_token_id
        self.dur_predictor = SeamlessM4TVariancePredictor(config)
        self.unit_embedding = nn.Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = nn.Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = nn.Embedding(config.vocoder_num_langs, config.lang_embed_dim)
        self.hifi_gan = SeamlessM4THifiGan(config)
        self.post_init()

    def _get_dur_output_lengths(self, input_ids, dur_out):
        """
        Computes the output length after the duration layer.
        """
        unit_lengths = (input_ids != self.pad_token_id).sum(1)
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()
        return unit_lengths

    def _get_output_hifigan_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the hifigan convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return torch.div(input_length + 2 * pad - dilation * (kernel_size - 1) - 1, stride, rounding_mode='floor') + 1

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        for i, (upsample_rate, kernel_size) in enumerate(zip(self.config.upsample_rates, self.config.upsample_kernel_sizes)):
            input_lengths = _transpose_conv_out_length(input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2)
        for i in range(len(self.config.upsample_rates)):
            for kernel_size, dilation in zip(self.config.resblock_kernel_sizes, self.config.resblock_dilation_sizes):
                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil)
                for dil in dilation:
                    input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        return input_lengths

    def forward(self, input_ids: torch.LongTensor, spkr_id: torch.Tensor, lang_id: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTextToUnitForConditionalGeneration`]. [What are input
                IDs?](../glossary#input-ids)
            spkr_id (`int`, *optional*):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            tgt_lang (`str`, *optional*):
                The language id to use as target language for translation.
        """
        hidden_states = self.unit_embedding(input_ids).transpose(1, 2)
        spkr = self.speaker_embedding(spkr_id).transpose(1, 2)
        lang = self.language_embedding(lang_id).transpose(1, 2)
        log_dur_pred = self.dur_predictor(hidden_states.transpose(1, 2))
        dur_out = torch.clamp(torch.round(torch.exp(log_dur_pred) - 1).long(), min=1)
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
        else:
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning('`self.training=True` and you use batching. You lose parallelism during the hifigan\n                               forward pass because the samples are interleaved.')
            hidden_states = [torch.repeat_interleave(hidden_state, duration, dim=-1).transpose(0, 1) for hidden_state, duration in zip(hidden_states, dur_out)]
            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True).transpose(1, 2)
        spkr = spkr.repeat(1, 1, hidden_states.shape[-1])
        lang = lang.repeat(1, 1, hidden_states.shape[-1])
        hidden_states = torch.cat([lang, hidden_states, spkr], dim=1)
        hidden_states = self.hifi_gan(hidden_states)
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out)
        lengths = self._get_output_hifigan_lengths(unit_lengths)
        return (hidden_states, lengths)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.hifi_gan.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.hifi_gan.conv_pre)
        for layer in self.hifi_gan.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.hifi_gan.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.hifi_gan.conv_post)