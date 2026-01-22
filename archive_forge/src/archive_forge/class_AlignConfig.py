import os
from typing import TYPE_CHECKING, List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class AlignConfig(PretrainedConfig):
    """
    [`AlignConfig`] is the configuration class to store the configuration of a [`AlignModel`]. It is used to
    instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AlignVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 640):
            Dimentionality of text and vision projection layers.
        temperature_init_value (`float`, *optional*, defaults to 1.0):
            The inital value of the *temperature* paramter. Default is used as per the original ALIGN implementation.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AlignConfig, AlignModel

    >>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignConfig()

    >>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
    >>> from transformers import AlignTextConfig, AlignVisionConfig

    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_text = AlignTextConfig()
    >>> config_vision = AlignVisionConfig()

    >>> config = AlignConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'align'

    def __init__(self, text_config=None, vision_config=None, projection_dim=640, temperature_init_value=1.0, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the AlignTextConfig with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the AlignVisionConfig with default values.')
        self.text_config = AlignTextConfig(**text_config)
        self.vision_config = AlignVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.temperature_init_value = temperature_init_value
        self.initializer_range = initializer_range

    @classmethod
    def from_text_vision_configs(cls, text_config: AlignTextConfig, vision_config: AlignVisionConfig, **kwargs):
        """
        Instantiate a [`AlignConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`AlignConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)