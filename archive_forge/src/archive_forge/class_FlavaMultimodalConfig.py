import os
from typing import Any, Dict, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class FlavaMultimodalConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FlavaMultimodalModel`]. It is used to instantiate
    an FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_cls_token (`bool`, *optional*, defaults to `True`):
            Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.


    Example:

    ```python
    >>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

    >>> # Initializing a FlavaMultimodalModel with  style configuration
    >>> configuration = FlavaMultimodalConfig()

    >>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
    >>> model = FlavaMultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'flava_multimodal_model'

    def __init__(self, hidden_size: int=768, num_hidden_layers: int=6, num_attention_heads: int=12, intermediate_size: int=3072, hidden_act: int='gelu', hidden_dropout_prob: int=0.0, attention_probs_dropout_prob: int=0.0, initializer_range: float=0.02, layer_norm_eps: float=1e-12, qkv_bias: bool=True, use_cls_token: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_cls_token = use_cls_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'flava':
            config_dict = config_dict['multimodal_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)