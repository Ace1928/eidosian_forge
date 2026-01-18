import logging
from typing import Any, Dict
import torch
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model, wavlm_model
Builds :class:`Wav2Vec2Model` from the corresponding model object of
    `Transformers <https://huggingface.co/transformers/>`_.

    Args:
        original (torch.nn.Module): An instance of ``Wav2Vec2ForCTC`` from ``transformers``.

    Returns:
        Wav2Vec2Model: Imported model.

    Example
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = import_huggingface_model(original)
        >>>
        >>> waveforms, _ = torchaudio.load("audio.wav")
        >>> logits, _ = model(waveforms)
    