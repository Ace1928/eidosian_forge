import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class Kosmos2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate a
    KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'kosmos-2'
    is_composition = True

    def __init__(self, text_config=None, vision_config=None, latent_query_num=64, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `Kosmos2TextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. Initializing the `Kosmos2VisionConfig` with default values.')
        self.text_config = Kosmos2TextConfig(**text_config)
        self.vision_config = Kosmos2VisionConfig(**vision_config)
        self.latent_query_num = latent_query_num