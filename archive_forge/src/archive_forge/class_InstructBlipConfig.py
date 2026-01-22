import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING
class InstructBlipConfig(PretrainedConfig):
    """
    [`InstructBlipConfig`] is the configuration class to store the configuration of a
    [`InstructBlipForConditionalGeneration`]. It is used to instantiate a InstructBLIP model according to the specified
    arguments, defining the vision model, Q-Former model and language model configs. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the InstructBLIP
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`InstructBlipQFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     InstructBlipVisionConfig,
    ...     InstructBlipQFormerConfig,
    ...     OPTConfig,
    ...     InstructBlipConfig,
    ...     InstructBlipForConditionalGeneration,
    ... )

    >>> # Initializing a InstructBlipConfig with Salesforce/instruct-blip-flan-t5 style configuration
    >>> configuration = InstructBlipConfig()

    >>> # Initializing a InstructBlipForConditionalGeneration (with random weights) from the Salesforce/instruct-blip-flan-t5 style configuration
    >>> model = InstructBlipForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a InstructBlipConfig from a InstructBlipVisionConfig, InstructBlipQFormerConfig and any PretrainedConfig

    >>> # Initializing InstructBLIP vision, InstructBLIP Q-Former and language model configurations
    >>> vision_config = InstructBlipVisionConfig()
    >>> qformer_config = InstructBlipQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = InstructBlipConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""
    model_type = 'instructblip'

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the InstructBlipVisionConfig with default values.')
        if qformer_config is None:
            qformer_config = {}
            logger.info('qformer_config is None. Initializing the InstructBlipQFormerConfig with default values.')
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the text config with default values (`OPTConfig`).')
        self.vision_config = InstructBlipVisionConfig(**vision_config)
        self.qformer_config = InstructBlipQFormerConfig(**qformer_config)
        text_model_type = text_config['model_type'] if 'model_type' in text_config else 'opt'
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder
        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(cls, vision_config: InstructBlipVisionConfig, qformer_config: InstructBlipQFormerConfig, text_config: PretrainedConfig, **kwargs):
        """
        Instantiate a [`InstructBlipConfig`] (or a derived class) from a InstructBLIP vision model, Q-Former and
        language model configurations.

        Returns:
            [`InstructBlipConfig`]: An instance of a configuration object
        """
        return cls(vision_config=vision_config.to_dict(), qformer_config=qformer_config.to_dict(), text_config=text_config.to_dict(), **kwargs)