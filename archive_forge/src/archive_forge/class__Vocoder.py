from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
class _Vocoder(ABC):

    @property
    @abstractmethod
    def sample_rate(self):
        """The sample rate of the resulting waveform

        :type: float
        """

    @abstractmethod
    def __call__(self, specgrams: Tensor, lengths: Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        """Generate waveform from the given input, such as spectrogram

        Args:
            specgrams (Tensor):
                The input spectrogram. Shape: `(batch, frequency bins, time)`.
                The expected shape depends on the implementation.
            lengths (Tensor, or None, optional):
                The valid length of each sample in the batch. Shape: `(batch, )`.
                (Default: `None`)

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor:
                The generated waveform. Shape: `(batch, max length)`
            Tensor or None:
                The valid length of each sample in the batch. Shape: `(batch, )`.
        """