import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class BridgeTowerVisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the vision configuration of a [`BridgeTowerModel`]. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in visual encoder model.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 288):
            The size (resolution) of each image.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        stop_gradient (`bool`, *optional*, defaults to `False`):
            Whether to stop gradient for training.
        share_layernorm (`bool`, *optional*, defaults to `True`):
            Whether LayerNorm layers are shared.
        remove_last_layer (`bool`, *optional*, defaults to `False`):
            Whether to remove the last layer from the vision encoder.


    Example:

    ```python
    >>> from transformers import BridgeTowerVisionConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the vision model
    >>> configuration = BridgeTowerVisionConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""
    model_type = 'bridgetower_vision_model'

    def __init__(self, hidden_size=768, num_hidden_layers=12, num_channels=3, patch_size=16, image_size=288, initializer_factor=1, layer_norm_eps=1e-05, stop_gradient=False, share_layernorm=True, remove_last_layer=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.stop_gradient = stop_gradient
        self.share_layernorm = share_layernorm
        self.remove_last_layer = remove_last_layer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'bridgetower':
            config_dict = config_dict['text_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)