from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
class MusicgenConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MusicgenModel`]. It is used to instantiate a
    MusicGen model according to the specified arguments, defining the text encoder, audio encoder and MusicGen decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
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
    ...     MusicgenConfig,
    ...     MusicgenDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenDecoderConfig()

    >>> configuration = MusicgenConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
    >>> model = MusicgenForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_config = MusicgenConfig.from_pretrained("musicgen-model")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("musicgen-model", config=musicgen_config)
    ```"""
    model_type = 'musicgen'
    is_composition = True

    def __init__(self, **kwargs):
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
        self.decoder = MusicgenDecoderConfig(**decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_sub_models_config(cls, text_encoder_config: PretrainedConfig, audio_encoder_config: PretrainedConfig, decoder_config: MusicgenDecoderConfig, **kwargs):
        """
        Instantiate a [`MusicgenConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """
        return cls(text_encoder=text_encoder_config.to_dict(), audio_encoder=audio_encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

    @property
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate