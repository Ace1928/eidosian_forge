import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
class ClapAudioEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        self.patch_embed = ClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins
        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // 2 ** i, grid_size[1] // 2 ** i) for i in range(self.num_layers)]
        self.layers = nn.ModuleList([ClapAudioStage(config=config, dim=int(config.patch_embeds_hidden_size * 2 ** i_layer), input_resolution=self.input_resolutions[i_layer], depth=config.depths[i_layer], num_heads=config.num_attention_heads[i_layer], drop_path=drop_path_rate[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=ClapAudioPatchMerging if i_layer < self.num_layers - 1 else None) for i_layer in range(self.num_layers)])
        self.gradient_checkpointing = False
        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        _, _, time_length, freq_length = normalized_input_features.shape
        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio
        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError('the wav size should be less than or equal to the swin input size')
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(normalized_input_features, (spec_width, freq_length), mode='bicubic', align_corners=True)
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(normalized_input_features, (time_length, spec_heigth), mode='bicubic', align_corners=True)
        batch, channels, time, freq = normalized_input_features.shape
        normalized_input_features = normalized_input_features.reshape(batch, channels * self.freq_ratio, time // self.freq_ratio, freq)
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        normalized_input_features = normalized_input_features.reshape(batch, channels, freq * self.freq_ratio, time // self.freq_ratio)
        return normalized_input_features

    def forward(self, input_features, is_longer: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, always_partition: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, ClapAudioModelOutput]:
        input_features = input_features.transpose(1, 3)
        normalized_input_features = self.batch_norm(input_features)
        normalized_input_features = normalized_input_features.transpose(1, 3)
        is_longer_list_idx = None
        if self.enable_fusion:
            is_longer_list = is_longer.to(input_features.device)
            is_longer_list_idx = torch.where(is_longer_list == 1)[0]
        hidden_states = self.reshape_mel2img(normalized_input_features)
        frames_num = hidden_states.shape[2]
        hidden_states = self.patch_embed(hidden_states, is_longer_list_idx)
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        input_dimensions = self.input_resolutions[0]
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            input_dimensions = self.input_resolutions[i]
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition)
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and (not output_hidden_states_before_downsampling):
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        last_hidden_state = self.norm(hidden_states)
        batch_size, _, n_channels = last_hidden_state.shape
        freq_shape = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[0]
        temporal_shape = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[1]
        last_hidden_state = last_hidden_state.permute(0, 2, 1).contiguous().reshape(batch_size, n_channels, freq_shape, temporal_shape)
        batch_size, n_channels, n_frequencies, n_temp = last_hidden_state.shape
        c_freq_bin = n_frequencies // self.freq_ratio
        last_hidden_state = last_hidden_state.reshape(batch_size, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp)
        last_hidden_state = last_hidden_state.permute(0, 1, 3, 2, 4).contiguous().reshape(batch_size, n_channels, c_freq_bin, -1)
        latent_output = self.avgpool(torch.flatten(last_hidden_state, 2))
        latent_output = torch.flatten(latent_output, 1)
        if not return_dict:
            return tuple((v for v in [last_hidden_state, latent_output, all_reshaped_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=latent_output, hidden_states=all_reshaped_hidden_states, attentions=all_self_attentions)