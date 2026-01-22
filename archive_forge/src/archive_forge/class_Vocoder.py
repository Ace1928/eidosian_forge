from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
class Vocoder(_Vocoder):
    """Interface of the vocoder part of Tacotron2TTS pipeline

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_vocoder` for the usage.
        """