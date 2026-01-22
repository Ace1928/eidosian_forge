from typing import Any, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging


    This is the configuration class to store the configuration of a [`DbrxModel`]. It is used to instantiate a Dbrx model according to the
    specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a different configuration to that of the [databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 2048):
            Dimensionality of the embeddings and hidden states.
        n_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        max_seq_len (`int`, *optional*, defaults to 2048):
            The maximum sequence length of the model.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`DbrxModel`].
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to the attention output before combining with residual.
        emb_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the embedding layer.
        attn_config (`dict`, *optional*):
            A dictionary used to configure the model's attention module.
        ffn_config (`dict`, *optional*):
            A dictionary used to configure the model's FFN module.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details.


    Example:
    ```python
    >>> from transformers import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig(n_layers=2, d_model=256, n_heads=8, vocab_size=128)

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    