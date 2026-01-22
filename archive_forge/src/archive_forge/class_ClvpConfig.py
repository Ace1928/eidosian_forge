import os
from typing import TYPE_CHECKING, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class ClvpConfig(PretrainedConfig):
    """
    [`ClvpConfig`] is the configuration class to store the configuration of a [`ClvpModelForConditionalGeneration`]. It
    is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
    decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the CLVP text encoder.
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize CLVP speech encoder.
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClvpDecoderConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimentionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLVP implementation.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

    >>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
    >>> configuration = ClvpConfig()

    >>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpModelForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

    >>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
    >>> config_text = ClvpEncoderConfig()
    >>> config_speech = ClvpEncoderConfig()
    >>> decoder_config = ClvpDecoderConfig()

    >>> config = ClvpConfig.from_sub_model_configs(config_text, config_speech, decoder_config)
    ```"""
    model_type = 'clvp'
    is_composition = True

    def __init__(self, text_config=None, speech_config=None, decoder_config=None, projection_dim=768, logit_scale_init_value=2.6592, initializer_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `ClvpEncoderConfig` with default values.')
        if speech_config is None:
            speech_config = {}
            logger.info('`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.')
        if decoder_config is None:
            decoder_config = {}
            logger.info('`decoder_config` is `None`. initializing the `ClvpDecoderConfig` with default values.')
        self.text_config = ClvpEncoderConfig(**text_config)
        self.speech_config = ClvpEncoderConfig(**speech_config)
        self.decoder_config = ClvpDecoderConfig(**decoder_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor

    @classmethod
    def from_sub_model_configs(cls, text_config: ClvpEncoderConfig, speech_config: ClvpEncoderConfig, decoder_config: ClvpDecoderConfig, **kwargs):
        """
        Instantiate a [`ClvpConfig`] (or a derived class) from CLVP text model configuration, CLVP speech model
        configuration and CLVP decoder model configuration.

        Args:
            text_config (`ClvpEncoderConfig`):
                Text model configuration of type [`ClvpEncoderConfig`].
            speech_config (`ClvpEncoderConfig`):
                Speech model configuration of type [`ClvpEncoderConfig`].
            decoder_config (`ClvpDecoderConfig`):
                Decoder model configuration of type [`ClvpDecoderConfig`].

        Returns:
            [`ClvpConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), speech_config=speech_config.to_dict(), decoder_config=decoder_config.to_dict(), **kwargs)