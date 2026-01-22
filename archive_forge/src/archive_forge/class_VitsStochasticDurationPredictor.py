import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsStochasticDurationPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size
        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(config, dropout_rate=config.duration_predictor_dropout)
        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)
        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))
        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(config, dropout_rate=config.duration_predictor_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, durations=None, reverse=False, noise_scale=1.0):
        inputs = torch.detach(inputs)
        inputs = self.conv_pre(inputs)
        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)
        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask
        if not reverse:
            hidden_states = self.post_conv_pre(durations)
            hidden_states = self.post_conv_dds(hidden_states, padding_mask)
            hidden_states = self.post_conv_proj(hidden_states) * padding_mask
            random_posterior = torch.randn(durations.size(0), 2, durations.size(2)).to(device=inputs.device, dtype=inputs.dtype) * padding_mask
            log_determinant_posterior_sum = 0
            latents_posterior = random_posterior
            for flow in self.post_flows:
                latents_posterior, log_determinant = flow(latents_posterior, padding_mask, global_conditioning=inputs + hidden_states)
                latents_posterior = torch.flip(latents_posterior, [1])
                log_determinant_posterior_sum += log_determinant
            first_half, second_half = torch.split(latents_posterior, [1, 1], dim=1)
            log_determinant_posterior_sum += torch.sum((nn.functional.logsigmoid(first_half) + nn.functional.logsigmoid(-first_half)) * padding_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + random_posterior ** 2) * padding_mask, [1, 2]) - log_determinant_posterior_sum
            first_half = (durations - torch.sigmoid(first_half)) * padding_mask
            first_half = torch.log(torch.clamp_min(first_half, 1e-05)) * padding_mask
            log_determinant_sum = torch.sum(-first_half, [1, 2])
            latents = torch.cat([first_half, second_half], dim=1)
            for flow in self.flows:
                latents, log_determinant = flow(latents, padding_mask, global_conditioning=inputs)
                latents = torch.flip(latents, [1])
                log_determinant_sum += log_determinant
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + latents ** 2) * padding_mask, [1, 2]) - log_determinant_sum
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            latents = torch.randn(inputs.size(0), 2, inputs.size(2)).to(device=inputs.device, dtype=inputs.dtype) * noise_scale
            for flow in flows:
                latents = torch.flip(latents, [1])
                latents, _ = flow(latents, padding_mask, global_conditioning=inputs, reverse=True)
            log_duration, _ = torch.split(latents, [1, 1], dim=1)
            return log_duration