import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class SiglipConfig(PretrainedConfig):
    """
    [`SiglipConfig`] is the configuration class to store the configuration of a [`SiglipModel`]. It is used to
    instantiate a Siglip model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SiglipVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import SiglipConfig, SiglipModel

    >>> # Initializing a SiglipConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipConfig()

    >>> # Initializing a SiglipModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SiglipConfig from a SiglipTextConfig and a SiglipVisionConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig

    >>> # Initializing a SiglipText and SiglipVision configuration
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()

    >>> config = SiglipConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'siglip'

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.')
        self.text_config = SiglipTextConfig(**text_config)
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: SiglipTextConfig, vision_config: SiglipVisionConfig, **kwargs):
        """
        Instantiate a [`SiglipConfig`] (or a derived class) from siglip text model configuration and siglip vision
        model configuration.

        Returns:
            [`SiglipConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)