from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..deprecated._archive_maps import MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402
class MusicgenMelodyConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MusicgenMelodyModel`]. It is used to instantiate a
    Musicgen Melody model according to the specified arguments, defining the text encoder, audio encoder and Musicgen Melody decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_chroma (`int`, *optional*, defaults to 12): Number of chroma bins to use.
        chroma_length (`int`, *optional*, defaults to 235):
            Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")
    >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)
    ```"""
    model_type = 'musicgen_melody'
    is_composition = True

    def __init__(self, num_chroma=12, chroma_length=235, **kwargs):
        super().__init__(**kwargs)
        if 'text_encoder' not in kwargs or 'audio_encoder' not in kwargs or 'decoder' not in kwargs:
            raise ValueError('Config has to be initialized with text_encoder, audio_encoder and decoder config')
        text_encoder_config = kwargs.pop('text_encoder')
        text_encoder_model_type = text_encoder_config.pop('model_type')
        audio_encoder_config = kwargs.pop('audio_encoder')
        audio_encoder_model_type = audio_encoder_config.pop('model_type')
        decoder_config = kwargs.pop('decoder')
        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = MusicgenMelodyDecoderConfig(**decoder_config)
        self.is_encoder_decoder = False
        self.num_chroma = num_chroma
        self.chroma_length = chroma_length

    @classmethod
    def from_sub_models_config(cls, text_encoder_config: PretrainedConfig, audio_encoder_config: PretrainedConfig, decoder_config: MusicgenMelodyDecoderConfig, **kwargs):
        """
        Instantiate a [`MusicgenMelodyConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenMelodyConfig`]: An instance of a configuration object
        """
        return cls(text_encoder=text_encoder_config.to_dict(), audio_encoder=audio_encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

    @property
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate

    @property
    def _attn_implementation(self):
        if hasattr(self, '_attn_implementation_internal'):
            if self._attn_implementation_internal is None:
                return 'eager'
            else:
                return self._attn_implementation_internal
        else:
            return 'eager'

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value
        self.decoder._attn_implementation = value