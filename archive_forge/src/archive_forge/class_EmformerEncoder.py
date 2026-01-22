from typing import List, Optional, Tuple
import torch
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.emformer import Emformer
from torchaudio.models.rnnt import _TimeReduction
class EmformerEncoder(torch.nn.Module):
    """Emformer Encoder class for HuBERT pre-training. Consists of emformer module,
        linear layer and layer normalization layer.

    Args:
        emformer (torch.nn.Module):
            :py:class:`torchaudio.models.Emformer` module that consists of a list of emformer layers.
        output_linear (torch.nn.Module):
            Linear layer after emformer module.
        layer_norm (torch.nn.Module):
            Apply layer normalization to the output.
    """

    def __init__(self, emformer: torch.nn.Module, output_linear: torch.nn.Module, layer_norm: torch.nn.Module):
        super().__init__()
        self.emformer = emformer
        self.output_linear = output_linear
        self.layer_norm = layer_norm

    def forward(self, input: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): The input feature for emformer encoder.
                Tensor with dimensions `(batch, time, feature_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.

        Returns:
            torch.Tensor: The feature Tensor after emformer encoder.
        """
        if lengths is None:
            B, T, _ = input.shape
            dummy_lengths = torch.full((B,), T)
            output, _ = self.emformer(input, dummy_lengths)
        else:
            output, lengths = self.emformer(input, lengths)
        output = self.output_linear(output)
        output = self.layer_norm(output)
        return output

    def extract_features(self, input: torch.Tensor, lengths: Optional[torch.Tensor], num_layers: Optional[int]=None) -> List[torch.Tensor]:
        """Extract output Tensors of the emformer layers.

        Args:
            input (torch.Tensor): The input feature for emformer encoder.
                Tensor with dimensions `(batch, time, feature_dim)`.
            lengths (torch.Tensor or None): Valid length of each input sample.
                Tensor with dimension `(batch, )`.
            num_layers (int or None, optional): If not ``None``, returns the first
                `num_layers` layers of Tensors as the output, otherwise returns the
                Tensors from all emformer layers.

        Returns:
            List[torch.Tensor]:
                Output Tensors of selected emformer layers.
        """
        if num_layers is not None:
            if not 0 < num_layers <= len(self.emformer.emformer_layers):
                raise ValueError(f'`num_layers` must be between [1, {len(self.emformer.emformer_layers)}]')
        ret: List[torch.Tensor] = []
        input = input.permute(1, 0, 2)
        right_context = self.emformer._gen_right_context(input)
        utterance = input[:input.size(0) - self.emformer.right_context_length]
        attention_mask = self.emformer._gen_attention_mask(utterance)
        mems = self.emformer.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:-1] if self.emformer.use_mem else torch.empty(0).to(dtype=input.dtype, device=input.device)
        output = utterance
        if lengths is None:
            B, T, _ = input.shape
            lengths = torch.full((B,), T)
        for layer in self.emformer.emformer_layers:
            output, right_context, mems = layer(output, lengths, right_context, mems, attention_mask)
            ret.append(output.permute(1, 0, 2))
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret