import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
class HuBERTPretrainModel(Module):
    """HuBERTPretrainModel()

    HuBERT model used for pretraining in *HuBERT* :cite:`hsu2021hubert`.

    Note:
        To build the model, please use one of the factory functions.

    See Also:
        `HuBERT Pre-training and Fine-tuning Recipes
        <https://github.com/pytorch/audio/tree/main/examples/hubert>`__

    Args:
        wav2vec2 (Wav2Vec2Model):
            Wav2Vec2 encoder that generates the transformer outputs.

        mask_generator (torch.nn.Module):
            Mask generator that generates the mask for masked prediction during the training.

        logit_generator (torch.nn.Module):
            Logit generator that predicts the logits of the masked and unmasked inputs.

        feature_grad_mult (float or None):
            The factor to scale the convolutional feature extraction layer gradients by.
            If ``None``, the gradients of feature extraction layers are not affected.
            The scale factor will not affect the forward pass.
    """

    def __init__(self, wav2vec2: Wav2Vec2Model, mask_generator: Module, logit_generator: Module, feature_grad_mult: Optional[float]):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.mask_generator = mask_generator
        self.logit_generator = logit_generator
        if feature_grad_mult is not None and (not 0.0 < feature_grad_mult < 1.0):
            raise ValueError(f'The value of `feature_grad_mult` must be ``None``or between (0, 1). Found {feature_grad_mult}')
        self.feature_grad_mult = feature_grad_mult

    def forward(self, waveforms: Tensor, labels: Tensor, audio_lengths: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of dimension `[batch, frames]`.
            labels (Tensor): Label for pre-training. A Tensor of dimension `[batch, frames]`.
            audio_lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `[batch, ]`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Tensor, Tensor):
            Tensor
                The masked sequences of probability distribution (in logit).
                Shape: `(masked_frames, num labels)`.
            Tensor
                The unmasked sequence of probability distribution (in logit).
                Shape: `(unmasked_frames, num labels)`.
            Tensor
                The feature mean value for additional penalty loss.
                Shape: `(1,)`.
        """
        x, lengths = self.wav2vec2.feature_extractor(waveforms, audio_lengths)
        if self.feature_grad_mult is not None and self.feature_grad_mult < 1.0:
            x = components.GradMultiply.apply(x, self.feature_grad_mult)
        features_pen = x.float().pow(2).mean()
        if lengths is not None:
            padding_mask = components._get_padding_mask(x, lengths)
        else:
            padding_mask = None
        x, attention_mask = self.wav2vec2.encoder._preprocess(x, lengths)
        x, mask = self.mask_generator(x, padding_mask)
        x = self.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        if x.shape[1] != labels.shape[1]:
            raise ValueError('The length of label must match that of HuBERT model output')
        if padding_mask is not None:
            mask_m = torch.logical_and(~padding_mask, mask)
            mask_u = torch.logical_and(~padding_mask, ~mask_m)
        else:
            mask_m = mask
            mask_u = ~mask_m
        logit_m, logit_u = self.logit_generator(x, labels, mask_m, mask_u)
        return (logit_m, logit_u, features_pen)