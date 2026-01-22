from typing import Any, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class DbrxAttentionConfig(PretrainedConfig):
    """Configuration class for Dbrx Attention.

    [`DbrxAttention`] class. It is used to instantiate attention layers
    according to the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        clip_qkv (`float`, *optional*):
            If set, clip the queries, keys, and values in the attention layer to this value.
        kv_n_heads (`Optional[int]`, defaults to 1): For grouped_query_attention only, allow user to specify number of kv heads.
        rope_theta (`float`, defaults to 10000.0): The base frequency for rope.
    """

    def __init__(self, attn_pdrop: float=0.0, clip_qkv: Optional[float]=None, kv_n_heads: int=1, rope_theta: float=10000.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.attn_pdrop = attn_pdrop
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta
        for k in ['model_type']:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f'Found unknown kwargs={kwargs!r}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'dbrx':
            config_dict = config_dict['attn_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type ' + f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)