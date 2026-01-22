import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class Pix2StructTextConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Pix2StructTextModel`]. It is used to instantiate
    a Pix2Struct text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Pix2Struct text decoder used by
    the [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50244):
            Vocabulary size of the `Pix2Struct` text model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`Pix2StructTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections in each attention head.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string).
        decoder_start_token_id (`int`, *optional*, defaults to 0):
            The id of the `decoder_start_token_id` token.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the `padding` token.
        eos_token_id (`int`, *optional*, defaults to 1):
            The id of the `end-of-sequence` token.

    Example:

    ```python
    >>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

    >>> # Initializing a Pix2StructTextConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructTextConfig()

    >>> # Initializing a Pix2StructTextModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'pix2struct_text_model'
    keys_to_ignore_at_inference = ['past_key_values']
    attribute_map = {'hidden_size': 'hidden_size', 'num_attention_heads': 'num_heads', 'num_hidden_layers': 'num_layers'}

    def __init__(self, vocab_size=50244, hidden_size=768, d_kv=64, d_ff=2048, num_layers=12, num_heads=12, relative_attention_num_buckets=32, relative_attention_max_distance=128, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, dense_act_fn='gelu_new', decoder_start_token_id=0, use_cache=False, pad_token_id=0, eos_token_id=1, tie_word_embeddings=False, is_decoder=True, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.dense_act_fn = dense_act_fn
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, decoder_start_token_id=decoder_start_token_id, tie_word_embeddings=tie_word_embeddings, is_decoder=is_decoder, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'pix2struct':
            config_dict = config_dict['text_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)