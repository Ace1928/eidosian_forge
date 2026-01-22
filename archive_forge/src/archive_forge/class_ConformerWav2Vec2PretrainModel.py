from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components
class ConformerWav2Vec2PretrainModel(Module):
    """Conformer Wav2Vec2 pre-train model for training from scratch.

    Note:
        To build the model, please use one of the factory functions,
        :py:func:`conformer_wav2vec2_base` or :py:func:`conformer_wav2vec2_large`

    Args:
        wav2vec2 (nn.Module):
            Conformer based Wav2Vec2 model, including feature extractor and conformer encoder components.
        mask_generator (nn.Module):
            Mask generator that generates the mask for masked prediction during training.
        negative_sampler (nn.Module):
            Negative sampler to apply after masking.

    """

    def __init__(self, wav2vec2: Wav2Vec2Model, mask_generator: Module, negative_sampler: Module):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.mask_generator = mask_generator
        self.negative_sampler = negative_sampler

    def forward(self, features: Tensor, audio_lengths: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Args:
            features (Tensor):
                Tensor of audio features of shape `(batch, frame, dim)`.
            audio_lengths (Tensor or None, optional):
                Tensor of valid length of each valid auidio in the batch.
                shape: `(batch, )` (Default: ``None``)

        Returns:
            (Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor):
            Tensor
                The masked sequences of probability distribution of shape `(batch, frame dim)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )` representing
                valid length in time axis is returns.
            Tensor
                The mask indices.
            Tensor
                The targets, prior to negative sampling.
            Tensor
                The negative samples.
            Tensor
                The indices of the negative samples.
        """
        x, lengths = self.wav2vec2.feature_extractor(features, audio_lengths)
        if lengths is not None:
            padding_mask = components._get_padding_mask(x, lengths)
        else:
            padding_mask = None
        x = self.wav2vec2.encoder.feature_projection.layer_norm(x)
        x = self.wav2vec2.encoder.feature_projection.dropout(x)
        unmasked_x = x.clone()
        x, mask_idxs = self.mask_generator(x, padding_mask)
        unmasked_x = unmasked_x[mask_idxs].view(x.shape[0], -1, x.shape[-1])
        targets, negs, neg_idxs = self.negative_sampler(unmasked_x)
        x = self.wav2vec2.encoder.feature_projection.projection(x)
        x = x.transpose(0, 1)
        for conformer_layer in self.wav2vec2.encoder.conformer:
            x = conformer_layer(x, padding_mask)
        x = x.transpose(0, 1)
        return (x, lengths, mask_idxs, targets, negs, neg_idxs)