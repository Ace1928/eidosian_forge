from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torch.nn import Module
from . import aligner, utils
@dataclass
class Wav2Vec2Bundle:
    """Data class that bundles associated information to use pretrained :py:class:`~torchaudio.models.Wav2Vec2Model`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Feature Extraction
        >>> import torchaudio
        >>>
        >>> bundle = torchaudio.pipelines.HUBERT_BASE
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 360M/360M [00:06<00:00, 60.6MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Extract acoustic features
        >>> features, _ = model.extract_features(waveform)
    """
    _path: str
    _params: Dict[str, Any]
    _sample_rate: float
    _normalize_waveform: bool
    _model_type: str

    @property
    def sample_rate(self) -> float:
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate

    def _get_state_dict(self, dl_kwargs):
        return utils._get_state_dict(self._path, dl_kwargs)

    def get_model(self, *, dl_kwargs=None) -> Module:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.Wav2Vec2Model`.

            For the models listed below, an additional layer normalization is performed on the input.

            For all other models, a :py:class:`~torchaudio.models.Wav2Vec2Model` instance is returned.

            - WAV2VEC2_LARGE_LV60K
            - WAV2VEC2_ASR_LARGE_LV60K_10M
            - WAV2VEC2_ASR_LARGE_LV60K_100H
            - WAV2VEC2_ASR_LARGE_LV60K_960H
            - WAV2VEC2_XLSR53
            - WAV2VEC2_XLSR_300M
            - WAV2VEC2_XLSR_1B
            - WAV2VEC2_XLSR_2B
            - HUBERT_LARGE
            - HUBERT_XLARGE
            - HUBERT_ASR_LARGE
            - HUBERT_ASR_XLARGE
            - WAVLM_LARGE
        """
        model = utils._get_model(self._model_type, self._params)
        state_dict = self._get_state_dict(dl_kwargs)
        model.load_state_dict(state_dict)
        if self._normalize_waveform:
            model = utils._extend_model(model, normalize_waveform=True)
        model.eval()
        return model