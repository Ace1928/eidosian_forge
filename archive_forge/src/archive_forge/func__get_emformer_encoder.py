from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
def _get_emformer_encoder(input_dim: int, output_dim: int, num_heads: int, ffn_dim: int, num_layers: int, segment_length: int, left_context_length: int, right_context_length: int, dropout: float, activation: str, max_memory_size: int, weight_init_scale_strategy: Optional[str], tanh_on_mem: bool) -> EmformerEncoder:
    """Construct EmformerEncoder for emformer model.

    Args:
        input_dim (int): The feature dimension of input Tensor.
        output_dim (int): The feature dimension after EmformerEncoder.
        num_heads (int): Number of attention heads in each Emformer layer.
        ffn_dim: (int): Hidden layer dimension of feedforward network.
        num_layers (int): Number of Emformer layers to instantiate.
        segment_length (int): Length of each input segment.
        left_context_length (int): Length of left context.
        right_context_length (int): Length of right context.
        dropout (float): Dropout probability.
        activation (str): Activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        max_memory_size (int): Maximum number of memory elements to use.
        weight_init_scale_strategy (str or None): Per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        tanh_on_mem (bool): If ``True``, applies tanh to memory elements.

    Returns:
        EmformerEncoder: The resulting EmformerEncoder module.
    """
    emformer = Emformer(input_dim=input_dim, num_heads=num_heads, ffn_dim=ffn_dim, num_layers=num_layers, segment_length=segment_length, left_context_length=left_context_length, right_context_length=right_context_length, dropout=dropout, activation=activation, max_memory_size=max_memory_size, weight_init_scale_strategy=weight_init_scale_strategy, tanh_on_mem=tanh_on_mem)
    output_linear = torch.nn.Linear(input_dim, output_dim)
    layer_norm = torch.nn.LayerNorm(output_dim)
    return EmformerEncoder(emformer, output_linear, layer_norm)