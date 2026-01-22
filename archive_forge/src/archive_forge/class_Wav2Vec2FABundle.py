from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torch.nn import Module
from . import aligner, utils
@dataclass
class Wav2Vec2FABundle(Wav2Vec2ASRBundle):
    """Data class that bundles associated information to use pretrained :py:class:`~torchaudio.models.Wav2Vec2Model` for forced alignment.

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
        >>> bundle = torchaudio.pipelines.MMS_FA
        >>>
        >>> # Build the model and load pretrained weight.
        >>> model = bundle.get_model()
        Downloading:
        100%|███████████████████████████████| 1.18G/1.18G [00:05<00:00, 216MB/s]
        >>>
        >>> # Resample audio to the expected sampling rate
        >>> waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        >>>
        >>> # Estimate the probability of token distribution
        >>> emission, _ = model(waveform)
        >>>
        >>> # Generate frame-wise alignment
        >>> alignment, scores = torchaudio.functional.forced_align(
        >>>     emission, targets, input_lengths, target_lengths, blank=0)
        >>>
    """

    class Tokenizer(aligner.ITokenizer):
        """Interface of the tokenizer"""

    class Aligner(aligner.IAligner):
        """Interface of the aligner"""

    def get_labels(self, star: Optional[str]='*', blank: str='-') -> Tuple[str, ...]:
        """Get the labels corresponding to the feature dimension of emission.

        The first is blank token, and it is customizable.

        Args:
            star (str or None, optional): Change or disable star token. (default: ``"*"``)
            blank (str, optional): Change the blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import MMS_FA as bundle
            >>> bundle.get_labels()
            ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x', '*')
            >>> bundle.get_labels(star=None)
            ('-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x')
        """
        labels = super().get_labels(blank=blank)
        return labels if star is None else (*labels, star)

    def get_model(self, with_star: bool=True, *, dl_kwargs=None) -> Module:
        """Construct the model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            with_star (bool, optional): If enabled, the last dimension of output layer is
                extended by one, which corresponds to `star` token.
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.models.Wav2Vec2Model`.

            .. note::

               The model created with this method returns probability in log-domain,
               (i.e. :py:func:`torch.nn.functional.log_softmax` is applied), whereas
               the other Wav2Vec2 models returns logit.
        """
        model = utils._get_model(self._model_type, self._params)
        state_dict = utils._get_state_dict(self._path, dl_kwargs, self._remove_aux_axis)
        model.load_state_dict(state_dict)
        model = utils._extend_model(model, normalize_waveform=self._normalize_waveform, apply_log_softmax=True, append_star=with_star)
        model.eval()
        return model

    def get_dict(self, star: Optional[str]='*', blank: str='-') -> Dict[str, int]:
        """Get the mapping from token to index (in emission feature dim)

        Args:
            star (str or None, optional): Change or disable star token. (default: ``"*"``)
            blank (str, optional): Change the blank token. (default: ``'-'``)

        Returns:
            Tuple[str, ...]:
            For models fine-tuned on ASR, returns the tuple of strings representing
            the output class labels.

        Example
            >>> from torchaudio.pipelines import MMS_FA as bundle
            >>> bundle.get_dict()
            {'-': 0, 'a': 1, 'i': 2, 'e': 3, 'n': 4, 'o': 5, 'u': 6, 't': 7, 's': 8, 'r': 9, 'm': 10, 'k': 11, 'l': 12, 'd': 13, 'g': 14, 'h': 15, 'y': 16, 'b': 17, 'p': 18, 'w': 19, 'c': 20, 'v': 21, 'j': 22, 'z': 23, 'f': 24, "'": 25, 'q': 26, 'x': 27, '*': 28}
            >>> bundle.get_dict(star=None)
            {'-': 0, 'a': 1, 'i': 2, 'e': 3, 'n': 4, 'o': 5, 'u': 6, 't': 7, 's': 8, 'r': 9, 'm': 10, 'k': 11, 'l': 12, 'd': 13, 'g': 14, 'h': 15, 'y': 16, 'b': 17, 'p': 18, 'w': 19, 'c': 20, 'v': 21, 'j': 22, 'z': 23, 'f': 24, "'": 25, 'q': 26, 'x': 27}
        """
        return {k: i for i, k in enumerate(self.get_labels(star=star, blank=blank))}

    def get_tokenizer(self) -> Tokenizer:
        """Instantiate a Tokenizer.

        Returns:
            Tokenizer
        """
        return aligner.Tokenizer(self.get_dict())

    def get_aligner(self) -> Aligner:
        """Instantiate an Aligner.

        Returns:
            Aligner
        """
        return aligner.Aligner(blank=0)