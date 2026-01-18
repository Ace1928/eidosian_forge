import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def wav2vec2_xlsr_1b(encoder_projection_dropout: float=0.1, encoder_attention_dropout: float=0.0, encoder_ff_interm_dropout: float=0.0, encoder_dropout: float=0.0, encoder_layer_drop: float=0.0, aux_num_out: Optional[int]=None) -> Wav2Vec2Model:
    """Builds XLS-R model :cite:`babu2021xls` with 1 billion of parameters. The architecture is compatible
    with Wav2Vec2 model :cite:`baevski2020wav2vec`, and so the output class is
    :class:`~torchaudio.models.Wav2Vec2Model`.

    Args:
        encoder_projection_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`~torchaudio.models.wav2vec2_model`.
        aux_num_out (int, optional):
            See :py:func:`~torchaudio.models.wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """
    return wav2vec2_model(extractor_mode='layer_norm', extractor_conv_layer_config=None, extractor_conv_bias=True, encoder_embed_dim=1280, encoder_projection_dropout=encoder_projection_dropout, encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16, encoder_num_layers=48, encoder_num_heads=16, encoder_attention_dropout=encoder_attention_dropout, encoder_ff_interm_features=5120, encoder_ff_interm_dropout=encoder_ff_interm_dropout, encoder_dropout=encoder_dropout, encoder_layer_norm_first=True, encoder_layer_drop=encoder_layer_drop, aux_num_out=aux_num_out)