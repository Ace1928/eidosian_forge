from typing import Any, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class DbrxFFNConfig(PretrainedConfig):
    """Configuration class for Dbrx FFN.

    [`DbrxFFN`] class. It is used to instantiate feedforward layers according to
    the specified arguments, defining the layers architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        ffn_act_fn (`dict`, *optional*, defaults to `None`): A dict specifying activation function for the FFN.
            The dict should have a key 'name' with the value being the name of the activation function along with
            any additional keyword arguments. If `None`, then set to `{"name": "silu"}`.
        ffn_hidden_size (`int`, defaults to 3584): The hidden size of the feedforward network.
        moe_num_experts (`int`, defaults to 4): The number of experts in the mixture of experts layer.
        moe_top_k (`int`, defaults to 1): The number of experts to use in the mixture of experts layer.
        moe_jitter_eps (`float`, *optional*, defaults to `None`): If not `None`, the jitter epsilon for the mixture of experts layer.
        moe_loss_weight (`float`, defaults to 0.01): The loss weight for the mixture of experts layer.
        moe_normalize_expert_weights (`float`, *optional*, defaults to 1.0): The normalization factor for the expert weights.
    """

    def __init__(self, ffn_act_fn: dict=None, ffn_hidden_size: int=3584, moe_num_experts: int=4, moe_top_k: int=1, moe_jitter_eps: Optional[float]=None, moe_loss_weight: float=0.01, moe_normalize_expert_weights: Optional[float]=1.0, **kwargs: Any):
        super().__init__()
        if ffn_act_fn is None:
            ffn_act_fn = {'name': 'silu'}
        self.ffn_act_fn = ffn_act_fn
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
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
            config_dict = config_dict['ffn_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type ' + f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)