import os
from typing import Any, Dict, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class FlavaConfig(PretrainedConfig):
    """
    [`FlavaConfig`] is the configuration class to store the configuration of a [`FlavaModel`]. It is used to
    instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
    and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaTextConfig`].
        image_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaImageConfig`].
        multimodal_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaMultimodalConfig`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and image projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original FLAVA/CLIP
            implementation.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        ce_ignore_index (`int`, *optional*, defaults to -100):
            Cross entropy index to ignore.
        mim_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MIM (Masked Image Modeling) unimodal loss
        mlm_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MLM (Masked Language Modeling) unimodal loss
        global_contrastive_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to global contrastive cross-alignment loss.
        itm_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to image-text matching multimodal loss.
        mmm_image_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MMM loss's image part.
        mmm_text_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MMM loss's text part.
        global_backprop_contrastive (`bool`, *optional*, defaults to `True`):
            Whether to use global backpropgation through all workers in contrastive loss.
        skip_unmasked_multimodal_encoder (`bool`, *optional*, defaults to `True`):
            Whether to skip running unmasked multimodal encoder whose outputs are not used by FLAVA losses.
        return_loss (`bool`, *optional*, defaults to `True`):
            Whether to return loss or not

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

    >>> # Initializing a FlavaConfig with style configuration
    >>> configuration = FlavaConfig()

    >>> # Initializing a FlavaModel and FlavaForPreTraining model (with random weights) from the style configuration
    >>> model = FlavaModel(configuration)
    >>> model_pre = FlavaForPreTraining(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> configuration_pre = model_pre.config
    ```
    """
    model_type = 'flava'

    def __init__(self, image_config: Dict[str, Any]=None, text_config: Dict[str, Any]=None, multimodal_config: Dict[str, Any]=None, image_codebook_config: Dict[str, Any]=None, hidden_size: int=768, layer_norm_eps: float=1e-12, projection_dim: int=768, init_codebook: bool=True, logit_scale_init_value: float=2.6592, initializer_range: float=0.02, ce_ignore_index: int=-100, mim_weight: float=1.0, mlm_weight: float=1.0, global_contrastive_weight: float=1.0, itm_weight: float=1.0, mmm_image_weight: float=1.0, mmm_text_weight: float=1.0, global_backprop_contrastive: bool=True, skip_unmasked_multimodal_encoder: bool=True, return_loss: bool=True, **kwargs):
        text_config_dict = kwargs.pop('text_config_dict', None)
        image_config_dict = kwargs.pop('image_config_dict', None)
        multimodal_config_dict = kwargs.pop('multimodal_config_dict', None)
        image_codebook_config_dict = kwargs.pop('image_codebook_config_dict', None)
        super().__init__(**kwargs)
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}
            _text_config_dict = FlavaTextConfig(**text_config_dict).to_dict()
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key not in ['transformers_version']):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `FlavaTextConfig`. The value `text_config["{key}"]` will be overriden.'
                    logger.info(message)
            text_config.update(_text_config_dict)
        if image_config_dict is not None:
            if image_config is None:
                image_config = {}
            _image_config_dict = FlavaImageConfig(**image_config_dict).to_dict()
            if 'id2label' in _image_config_dict:
                _image_config_dict['id2label'] = {str(key): value for key, value in _image_config_dict['id2label'].items()}
            for key, value in _image_config_dict.items():
                if key in image_config and value != image_config[key] and (key not in ['transformers_version']):
                    if key in image_config_dict:
                        message = f'`{key}` is found in both `image_config_dict` and `image_config` but with different values. The value `image_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`image_config_dict` is provided which will be used to initialize `FlavaImageConfig`. The value `image_config["{key}"]` will be overriden.'
                    logger.info(message)
            image_config.update(_image_config_dict)
        if multimodal_config_dict is not None:
            if multimodal_config is None:
                multimodal_config = {}
            _multimodal_config_dict = FlavaMultimodalConfig(**multimodal_config_dict).to_dict()
            for key, value in _multimodal_config_dict.items():
                if key in multimodal_config and value != multimodal_config[key] and (key not in ['transformers_version']):
                    if key in multimodal_config_dict:
                        message = f'`{key}` is found in both `multimodal_config_dict` and `multimodal_config` but with different values. The value `multimodal_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`multimodal_config_dict` is provided which will be used to initialize `FlavaMultimodalConfig`. The value `multimodal_config["{key}"]` will be overriden.'
                    logger.info(message)
            multimodal_config.update(_multimodal_config_dict)
        if image_codebook_config_dict is not None:
            if image_codebook_config is None:
                image_codebook_config = {}
            _image_codebook_config_dict = FlavaImageCodebookConfig(**image_codebook_config_dict).to_dict()
            for key, value in _image_codebook_config_dict.items():
                if key in image_codebook_config and value != image_codebook_config[key] and (key not in ['transformers_version']):
                    if key in image_codebook_config_dict:
                        message = f'`{key}` is found in both `image_codebook_config_dict` and `image_codebook_config` but with different values. The value `image_codebook_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`image_codebook_config_dict` is provided which will be used to initialize `FlavaImageCodebookConfig`. The value `image_codebook_config["{key}"]` will be overriden.'
                    logger.info(message)
            image_codebook_config.update(_image_codebook_config_dict)
        if image_config is None:
            image_config = {}
            logger.info('`image_config` is `None`. initializing the `FlavaImageConfig` with default values.')
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `FlavaTextConfig` with default values.')
        if multimodal_config is None:
            multimodal_config = {}
            logger.info('`multimodal_config` is `None`. initializing the `FlavaMultimodalConfig` with default values.')
        if image_codebook_config is None:
            image_codebook_config = {}
            logger.info('`image_codebook_config` is `None`. initializing the `FlavaImageCodebookConfig` with default values.')
        self.image_config = FlavaImageConfig(**image_config)
        self.text_config = FlavaTextConfig(**text_config)
        self.multimodal_config = FlavaMultimodalConfig(**multimodal_config)
        self.image_codebook_config = FlavaImageCodebookConfig(**image_codebook_config)
        self.projection_dim = projection_dim
        self.init_codebook = init_codebook
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.ce_ignore_index = ce_ignore_index
        self.mim_weight = mim_weight
        self.mlm_weight = mlm_weight
        self.global_contrastive_weight = global_contrastive_weight
        self.itm_weight = itm_weight
        self.mmm_image_weight = mmm_image_weight
        self.mmm_text_weight = mmm_text_weight
        self.global_backprop_contrastive = global_backprop_contrastive
        self.skip_unmasked_multimodal_encoder = skip_unmasked_multimodal_encoder
        self.return_loss = return_loss

    @classmethod
    def from_configs(cls, image_config: FlavaImageConfig, text_config: FlavaTextConfig, multimodal_config: FlavaMultimodalConfig, image_codebook_config: FlavaImageCodebookConfig, **kwargs):
        """
        Instantiate a [`FlavaConfig`] (or a derived class) from flava text model configuration, flava image model
        configuration, flava multimodal model and flava codebook model configuration.

        Returns:
            [`FlavaConfig`]: An instance of a configuration object
        """
        return cls(image_config=image_config.to_dict(), text_config=text_config.to_dict(), multimodal_config=multimodal_config.to_dict(), image_codebook_config=image_codebook_config.to_dict(), **kwargs)