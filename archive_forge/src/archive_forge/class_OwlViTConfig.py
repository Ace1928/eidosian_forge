import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class OwlViTConfig(PretrainedConfig):
    """
    [`OwlViTConfig`] is the configuration class to store the configuration of an [`OwlViTModel`]. It is used to
    instantiate an OWL-ViT model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWL-ViT
    [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OwlViTVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original OWL-ViT
            implementation.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a dictionary. If `False`, returns a tuple.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'owlvit'

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, return_dict=True, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the OwlViTTextConfig with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the OwlViTVisionConfig with default values.')
        self.text_config = OwlViTTextConfig(**text_config)
        self.vision_config = OwlViTVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
        """
        Instantiate a [`OwlViTConfig`] (or a derived class) from owlvit text model configuration and owlvit vision
        model configuration.

        Returns:
            [`OwlViTConfig`]: An instance of a configuration object
        """
        config_dict = {}
        config_dict['text_config'] = text_config
        config_dict['vision_config'] = vision_config
        return cls.from_dict(config_dict, **kwargs)