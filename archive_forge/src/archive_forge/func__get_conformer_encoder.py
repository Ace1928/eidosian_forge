from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
def _get_conformer_encoder(in_features: int, embed_dim: int, dropout_input: float, num_layers: int, num_heads: int, ff_interm_features: int, dropout: float, depthwise_conv_kernel_size: Union[int, List[int]], convolution_first: bool, use_group_norm: bool) -> ConformerEncoder:
    """Construct Conformer Encoder

    Args:
        in_features (int): The number of input features.
        embed_dim (int): The dimension of the embedding in the feature projection.
        dropout_input (float): The dropout probability applied after the input feature
            is projected to ``embed_dim``.
        num_layers (int): Number of Conformer layers in the encoder.
        num_heads (int): Number of heads in each Conformer layer.
        ff_interm_features (int): Hidden layer dimension of the feedforward network in
            each Conformer layer.
        dropout (float): Dropout probability in each Conformer layer.
        depthwise_conv_kernel_size (int or List[int]): List of kernel sizes corresponding
            to each of the  Conformer layers.If int is provided, all layers will have the
            same kernel size.
        convolution_first (bool): Whether to apply the convolution module ahead of the
            attention module in each Conformer layer.
        use_group_norm (bool): Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in
            the convolution module in each Conformer layer.

    Returns:
        ConformerEncoder:
            The resulting conformer encoder module.
    """
    feature_projection = components.FeatureProjection(in_features, embed_dim, dropout_input)
    if type(depthwise_conv_kernel_size) == int:
        depthwise_conv_kernel_size = [depthwise_conv_kernel_size] * num_layers
    assert len(depthwise_conv_kernel_size) == num_layers
    conformer_layers = []
    for l in range(num_layers):
        layer = ConformerLayer(input_dim=embed_dim, ffn_dim=ff_interm_features, num_attention_heads=num_heads, depthwise_conv_kernel_size=depthwise_conv_kernel_size[l], dropout=dropout, use_group_norm=use_group_norm, convolution_first=convolution_first)
        conformer_layers.append(layer)
    return ConformerEncoder(feature_projection, ModuleList(conformer_layers))