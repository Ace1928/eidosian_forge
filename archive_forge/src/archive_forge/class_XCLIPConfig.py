import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class XCLIPConfig(PretrainedConfig):
    """
    [`XCLIPConfig`] is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to
    instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        prompt_layers (`int`, *optional*, defaults to 2):
            Number of layers in the video specific prompt generator.
        prompt_alpha (`float`, *optional*, defaults to 0.1):
            Alpha value to use in the video specific prompt generator.
        prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the video specific prompt generator. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        prompt_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the cross-attention of the video specific prompt generator.
        prompt_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers in the video specific prompt generator.
        prompt_projection_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the projection layers in the video specific prompt generator.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original XCLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'xclip'

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, prompt_layers=2, prompt_alpha=0.1, prompt_hidden_act='quick_gelu', prompt_num_attention_heads=8, prompt_attention_dropout=0.0, prompt_projection_dropout=0.0, logit_scale_init_value=2.6592, **kwargs):
        text_config_dict = kwargs.pop('text_config_dict', None)
        vision_config_dict = kwargs.pop('vision_config_dict', None)
        super().__init__(**kwargs)
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}
            _text_config_dict = XCLIPTextConfig(**text_config_dict).to_dict()
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key not in ['transformers_version']):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `XCLIPTextConfig`. The value `text_config["{key}"]` will be overriden.'
                    logger.info(message)
            text_config.update(_text_config_dict)
        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}
            _vision_config_dict = XCLIPVisionConfig(**vision_config_dict).to_dict()
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {str(key): value for key, value in _vision_config_dict['id2label'].items()}
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and (key not in ['transformers_version']):
                    if key in vision_config_dict:
                        message = f'`{key}` is found in both `vision_config_dict` and `vision_config` but with different values. The value `vision_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`vision_config_dict` is provided which will be used to initialize `XCLIPVisionConfig`. The value `vision_config["{key}"]` will be overriden.'
                    logger.info(message)
            vision_config.update(_vision_config_dict)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `XCLIPTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `XCLIPVisionConfig` with default values.')
        self.text_config = XCLIPTextConfig(**text_config)
        self.vision_config = XCLIPVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.prompt_layers = prompt_layers
        self.prompt_alpha = prompt_alpha
        self.prompt_hidden_act = prompt_hidden_act
        self.prompt_num_attention_heads = prompt_num_attention_heads
        self.prompt_attention_dropout = prompt_attention_dropout
        self.prompt_projection_dropout = prompt_projection_dropout
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: XCLIPTextConfig, vision_config: XCLIPVisionConfig, **kwargs):
        """
        Instantiate a [`XCLIPConfig`] (or a derived class) from xclip text model configuration and xclip vision model
        configuration.

        Returns:
            [`XCLIPConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)