import math
from typing import Dict, List, Optional, Tuple
import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber
class _ConformerEncoder(torch.nn.Module, _Transcriber):

    def __init__(self, *, input_dim: int, output_dim: int, time_reduction_stride: int, conformer_input_dim: int, conformer_ffn_dim: int, conformer_num_layers: int, conformer_num_heads: int, conformer_depthwise_conv_kernel_size: int, conformer_dropout: float) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = torch.nn.Linear(input_dim * time_reduction_stride, conformer_input_dim)
        self.conformer = Conformer(num_layers=conformer_num_layers, input_dim=conformer_input_dim, ffn_dim=conformer_ffn_dim, num_heads=conformer_num_heads, depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size, dropout=conformer_dropout, use_group_norm=True, convolution_first=True)
        self.output_linear = torch.nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return (layer_norm_out, lengths)

    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states: Optional[List[List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError('Conformer does not support streaming inference.')