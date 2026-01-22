from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchaudio._internal import load_state_dict_from_url
from torchaudio.prototype.models.hifi_gan import hifigan_vocoder, HiFiGANVocoder
from torchaudio.transforms import MelSpectrogram
@dataclass
class HiFiGANVocoderBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.prototype.models.HiFiGANVocoder`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    This bundle can convert mel spectrorgam to waveforms and vice versa. A typical use case would be a flow like
    `text -> mel spectrogram -> waveform`, where one can use an external component, e.g. Tacotron2,
    to generate mel spectrogram from text. Please see below for the code example.

    Example: Transform synthetic mel spectrogram to audio.
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle
        >>>
        >>> # Load the HiFiGAN bundle
        >>> vocoder = bundle.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_vocoder_v3_ljspeech.pth"
        100%|████████████| 5.59M/5.59M [00:00<00:00, 18.7MB/s]
        >>>
        >>> # Generate synthetic mel spectrogram
        >>> specgram = torch.sin(0.5 * torch.arange(start=0, end=100)).expand(bundle._vocoder_params["in_channels"], 100)
        >>>
        >>> # Transform mel spectrogram into audio
        >>> waveform = vocoder(specgram)
        >>> torchaudio.save('sample.wav', waveform, bundle.sample_rate)

    Example: Usage together with Tacotron2, text to audio.
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle_hifigan
        >>>
        >>> # Load Tacotron2 bundle
        >>> bundle_tactron2 = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        >>> processor = bundle_tactron2.get_text_processor()
        >>> tacotron2 = bundle_tactron2.get_tacotron2()
        >>>
        >>> # Use Tacotron2 to convert text to mel spectrogram
        >>> text = "A quick brown fox jumped over a lazy dog"
        >>> input, lengths = processor(text)
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> # Load HiFiGAN bundle
        >>> vocoder = bundle_hifigan.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_vocoder_v3_ljspeech.pth"
        100%|████████████| 5.59M/5.59M [00:03<00:00, 1.55MB/s]
        >>>
        >>> # Use HiFiGAN to convert mel spectrogram to audio
        >>> waveform = vocoder(specgram).squeeze(0)
        >>> torchaudio.save('sample.wav', waveform, bundle_hifigan.sample_rate)
    """
    _path: str
    _vocoder_params: Dict[str, Any]
    _mel_params: Dict[str, Any]
    _sample_rate: float

    def _get_state_dict(self, dl_kwargs):
        url = f'https://download.pytorch.org/torchaudio/models/{self._path}'
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_vocoder(self, *, dl_kwargs=None) -> HiFiGANVocoder:
        """Construct the HiFiGAN Generator model, which can be used a vocoder, and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.prototype.models.HiFiGANVocoder`.
        """
        model = hifigan_vocoder(**self._vocoder_params)
        model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.eval()
        return model

    def get_mel_transform(self) -> Module:
        """Construct an object which transforms waveforms into mel spectrograms."""
        return _HiFiGANMelSpectrogram(n_mels=self._vocoder_params['in_channels'], sample_rate=self._sample_rate, **self._mel_params)

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate