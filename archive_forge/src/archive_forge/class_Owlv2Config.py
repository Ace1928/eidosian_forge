import os
from typing import TYPE_CHECKING, Dict, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class Owlv2Config(PretrainedConfig):
    """
    [`Owlv2Config`] is the configuration class to store the configuration of an [`Owlv2Model`]. It is used to
    instantiate an OWLv2 model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWLv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Owlv2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Owlv2VisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original OWLv2
            implementation.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a dictionary. If `False`, returns a tuple.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'owlv2'

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, return_dict=True, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the Owlv2TextConfig with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the Owlv2VisionConfig with default values.')
        self.text_config = Owlv2TextConfig(**text_config)
        self.vision_config = Owlv2VisionConfig(**vision_config)
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
        Instantiate a [`Owlv2Config`] (or a derived class) from owlv2 text model configuration and owlv2 vision
        model configuration.

        Returns:
            [`Owlv2Config`]: An instance of a configuration object
        """
        config_dict = {}
        config_dict['text_config'] = text_config
        config_dict['vision_config'] = vision_config
        return cls.from_dict(config_dict, **kwargs)