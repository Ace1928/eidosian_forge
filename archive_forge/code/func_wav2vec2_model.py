import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def wav2vec2_model(extractor_mode: str, extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]], extractor_conv_bias: bool, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_pos_conv_kernel: int, encoder_pos_conv_groups: int, encoder_num_layers: int, encoder_num_heads: int, encoder_attention_dropout: float, encoder_ff_interm_features: int, encoder_ff_interm_dropout: float, encoder_dropout: float, encoder_layer_norm_first: bool, encoder_layer_drop: float, aux_num_out: Optional[int]) -> Wav2Vec2Model:
    """Builds custom :class:`~torchaudio.models.Wav2Vec2Model`.

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        :cite:`baevski2020wav2vec` paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.
        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    feature_extractor = components._get_feature_extractor(extractor_mode, extractor_conv_layer_config, extractor_conv_bias)
    encoder = components._get_encoder(in_features=extractor_conv_layer_config[-1][0], embed_dim=encoder_embed_dim, dropout_input=encoder_projection_dropout, pos_conv_kernel=encoder_pos_conv_kernel, pos_conv_groups=encoder_pos_conv_groups, num_layers=encoder_num_layers, num_heads=encoder_num_heads, attention_dropout=encoder_attention_dropout, ff_interm_features=encoder_ff_interm_features, ff_interm_dropout=encoder_ff_interm_dropout, dropout=encoder_dropout, layer_norm_first=encoder_layer_norm_first, layer_drop=encoder_layer_drop)
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, encoder, aux)