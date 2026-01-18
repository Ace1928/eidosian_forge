"""
🔥😈 Qwen2 Model Architecture: Eidos's Modular Design

This module embodies Eidos's approach to constructing Qwen2-like models, emphasizing modularity, flexibility, and robustness.
It provides a factory-based system for creating models with customizable configurations, allowing for experimentation and adaptation.

Core Components:

- **Token Embedding Layer**: Converts discrete tokens into continuous vector representations.
- **Qwen2DecoderLayer**: The fundamental building block, repeated N times, comprising:
    - **Self-Attention Mechanism (Qwen2SdpaAttention)**:
        - **Query Projection (q_proj)**: Linear transformation of input to query vectors.
        - **Key Projection (k_proj)**: Linear transformation of input to key vectors.
        - **Value Projection (v_proj)**: Linear transformation of input to value vectors.
        - **Output Projection (o_proj)**: Linear transformation of attention output.
        - **Rotary Embeddings (rotary_emb)**: Positional encoding for self-attention.
    - **Multilayer Perceptron (Qwen2MLP)**:
        - **Gate Projection (gate_proj)**: Linear transformation for gating mechanism.
        - **Up Projection (up_proj)**: Linear transformation to expand dimensionality.
        - **Down Projection (down_proj)**: Linear transformation to reduce dimensionality.
        - **Activation Function (act_fn)**: SiLU activation for non-linearity.
    - **Input Layer Normalization (input_layernorm)**: RMSNorm applied before self-attention.
    - **Post-Attention Layer Normalization (post_attention_layernorm)**: RMSNorm applied after self-attention and MLP.
- **Final Layer Normalization (norm)**: RMSNorm applied to the output of the final decoder layer.
- **Language Model Head (lm_head)**: Linear transformation to project hidden states to vocabulary space.

Eidosian Principles:

- **Modularity**: Each component is designed as a standalone module, promoting reusability and maintainability.
- **Flexibility**: The factory pattern allows for easy customization of model configurations.
- **Robustness**: Error handling and default values ensure stable operation.
- **Adaptability**: The design supports future extensions and modifications.
- **Parameterization**: All configurable options are parameterized with defaults.

Potential Improvements and Next-Generation Features (Eidosian Vision):

1.  **Mixture of Experts (MoE) Layers**: Integrate MoE layers within the decoder stack to increase model capacity without a proportional increase in computational cost. This would allow the model to specialize in different aspects of the input data.
2.  **Adaptive Attention Span**: Implement mechanisms for the attention mechanism to dynamically adjust its span based on the input sequence, improving efficiency and handling long-range dependencies more effectively.
3.  **Gated Linear Units (GLU) in MLP**: Replace the current MLP with a GLU-based MLP to potentially improve the model's ability to capture complex relationships in the data.
4.  **Flash Attention**: Implement Flash Attention for faster and more memory-efficient attention computations, especially beneficial for long sequences.
5.  **Quantization and Pruning**: Apply quantization and pruning techniques to reduce the model's size and computational requirements without significant loss in performance.
6.  **Neural Architecture Search (NAS)**: Employ NAS to automatically discover optimal layer configurations and hyperparameters for the model.
7.  **Continuous Learning**: Implement a continuous learning framework to allow the model to adapt to new data and tasks without catastrophic forgetting.
8.  **Specialized Layers**: Introduce specialized layers for specific tasks, such as layers for handling code, mathematical reasoning, or multimodal inputs.
9.  **Hardware-Aware Optimizations**: Optimize the model architecture for specific hardware platforms, such as GPUs or TPUs, to maximize performance.
10. **Hierarchical Attention**: Implement hierarchical attention mechanisms to capture both local and global dependencies in the input sequence.

Qwen2 Model Factory:

This factory will allow the construction of arbitrary Qwen2-like models using modular components.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Union
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    Qwen2SdpaAttention,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2Model,
    Qwen2Config,
)
from transformers import AutoConfig, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

# [all]
# Config
# Qwen2LayerFactory
# Qwen2LayerFactory.__init__
# Qwen2LayerFactory.create_rotary_embedding
# Qwen2LayerFactory.create_attention
# Qwen2LayerFactory.create_mlp
# Qwen2LayerFactory.create_rms_norm
# Qwen2LayerFactory.create_decoder_layer
# Qwen2LayerFactory.create_embedding_layer
# Qwen2LayerFactory.create_lm_head
# Qwen2DecoderLayer
# Qwen2DecoderLayer.__init__
# Qwen2DecoderLayer.forward
# Qwen2ModelFactory
# Qwen2ModelFactory.__init__
# Qwen2ModelFactory.create_model


class Config(Qwen2Config):
    """
    🔥😈 Configuration class for Qwen2 model, extending the base Qwen2Config from Hugging Face Transformers.
    This class provides default values and allows for customization of the model's architecture.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 0.000001,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000,
        rope_scaling: Optional[Any] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0,
        **kwargs: Any,
    ):
        """
        🔥😈 Initializes the configuration with specified or default values.

        Args:
            vocab_size (int): The vocabulary size of the model. Defaults to 151936.
            hidden_size (int): The hidden size of the model. Defaults to 4096.
            intermediate_size (int): The intermediate size of the MLP layers. Defaults to 22016.
            num_hidden_layers (int): The number of hidden layers in the model. Defaults to 32.
            num_attention_heads (int): The number of attention heads. Defaults to 32.
            num_key_value_heads (int): The number of key/value heads. Defaults to 32.
            hidden_act (str): The activation function for the MLP layers. Defaults to "silu".
            max_position_embeddings (int): The maximum sequence length. Defaults to 32768.
            initializer_range (float): The range for initializing weights. Defaults to 0.02.
            rms_norm_eps (float): The epsilon value for RMS normalization. Defaults to 0.000001.
            use_cache (bool): Whether to use key/value caching. Defaults to True.
            tie_word_embeddings (bool): Whether to tie word embeddings. Defaults to False.
            rope_theta (float): The theta value for RoPE embeddings. Defaults to 10000.
            rope_scaling (Optional[Any]): The scaling configuration for RoPE. Defaults to None.
            use_sliding_window (bool): Whether to use sliding window attention. Defaults to False.
            sliding_window (int): The size of the sliding window. Defaults to 4096.
            max_window_layers (int): The number of layers to apply sliding window attention. Defaults to 28.
            attention_dropout (float): The dropout probability for attention. Defaults to 0.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            **kwargs,
        )


class Qwen2LayerFactory:
    """
    🔥😈 A factory class to create Qwen2 model components, embodying Eidos's modular design principles.
    This factory is responsible for creating individual layers of the Qwen2 model, such as rotary embeddings, attention layers,
    MLP layers, and normalization layers. It uses a configuration object to determine the parameters of each layer.
    """

    def __init__(self, config: AutoConfig):
        """
        🔥😈 Initializes the layer factory with the given configuration.

        Args:
            config (AutoConfig): The configuration for the Qwen2 model, loaded from a pretrained model or a custom config.
        """
        try:
            self.config = config
            # Extracting configuration parameters with default values
            self.hidden_size = getattr(config, "hidden_size", 768)  # Default to 768
            self.num_attention_heads = getattr(
                config, "num_attention_heads", 12
            )  # Default to 12
            self.intermediate_size = getattr(
                config, "intermediate_size", 3072
            )  # Default to 3072
            self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)  # Default to 1e-6
            self.vocab_size = getattr(config, "vocab_size", 50257)  # Default to 50257
            self.attention_dropout = getattr(
                config, "attention_dropout", 0.0
            )  # Default to 0.0
            self.rope_scaling = getattr(config, "rope_scaling", None)  # Default to None
            self.use_flash_attention_2 = getattr(
                config, "use_flash_attention_2", False
            )  # Default to False
            self.hidden_act = getattr(config, "hidden_act", "silu")  # Default to silu
            logger.info("😈🔥 Eidos: Qwen2LayerFactory initialized with config.")
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error initializing Qwen2LayerFactory: {e}", exc_info=True
            )
            raise

    def create_rotary_embedding(
        self, hidden_size: Optional[int] = None
    ) -> Qwen2RotaryEmbedding:
        """
        🔥😈 Creates a rotary embedding layer.

        Args:
            hidden_size (Optional[int]): The hidden size of the model. Defaults to the factory's hidden size if None.

        Returns:
            Qwen2RotaryEmbedding: The rotary embedding layer.
        """
        try:
            # Use provided hidden_size or default to factory's hidden_size
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            logger.debug(f"🌀 Eidos: Creating rotary embedding with hidden_size: {hs}")
            return Qwen2RotaryEmbedding(hs)
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error creating rotary embedding: {e}", exc_info=True
            )
            raise

    def create_attention(
        self,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        attention_dropout: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_flash_attention_2: Optional[bool] = None,
        layer_idx: Optional[int] = None,
    ) -> Qwen2SdpaAttention:
        """
        🔥😈 Creates a self-attention layer.

        Args:
            hidden_size (Optional[int]): The hidden size of the model. Defaults to the factory's hidden size if None.
            num_attention_heads (Optional[int]): The number of attention heads. Defaults to the factory's num_attention_heads if None.
            attention_dropout (Optional[float]): The dropout probability for attention. Defaults to the factory's attention_dropout if None.
            rope_scaling (Optional[Dict[str, Any]]): The rope scaling configuration. Defaults to the factory's rope_scaling if None.
            use_flash_attention_2 (Optional[bool]): Whether to use flash attention 2. Defaults to the factory's use_flash_attention_2 if None.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            Qwen2SdpaAttention: The self-attention layer.
        """
        try:
            # Use provided values or default to factory's values
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            nah = (
                num_attention_heads
                if num_attention_heads is not None
                else self.num_attention_heads
            )
            if nah is None:
                nah = 12  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: num_attention_heads not provided, using fallback default: {nah}"
                )
            ad = (
                attention_dropout
                if attention_dropout is not None
                else self.attention_dropout
            )
            if ad is None:
                ad = 0.0  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: attention_dropout not provided, using fallback default: {ad}"
                )
            rs = rope_scaling if rope_scaling is not None else self.rope_scaling
            ufa = (
                use_flash_attention_2
                if use_flash_attention_2 is not None
                else self.use_flash_attention_2
            )
            if ufa is None:
                ufa = False  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: use_flash_attention_2 not provided, using fallback default: {ufa}"
                )
            logger.debug(
                f"🌀 Eidos: Creating attention layer with hidden_size: {hs}, num_attention_heads: {nah}, attention_dropout: {ad}, rope_scaling: {rs}, use_flash_attention_2: {ufa}, layer_idx: {layer_idx}"
            )

            # Create a dummy config for Qwen2SdpaAttention
            config = Qwen2Config(
                hidden_size=hs,
                num_attention_heads=nah,
                attention_dropout=ad,
                rope_scaling=rs,
                use_flash_attention_2=ufa,
                layer_idx=layer_idx,
            )

            return Qwen2SdpaAttention(config=config)
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error creating attention layer: {e}", exc_info=True
            )
            raise

    def create_mlp(
        self,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ) -> Qwen2MLP:
        """
        🔥😈 Creates a multi-layer perceptron (MLP) layer.

        Args:
            hidden_size (Optional[int]): The hidden size of the model. Defaults to the factory's hidden size if None.
            intermediate_size (Optional[int]): The intermediate size of the MLP. Defaults to the factory's intermediate_size if None.
            hidden_act (Optional[str]): The activation function for the MLP. Defaults to the factory's hidden_act if None.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            Qwen2MLP: The MLP layer.
        """
        try:
            # Use provided values or default to factory's values
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            isize = (
                intermediate_size
                if intermediate_size is not None
                else self.intermediate_size
            )
            if isize is None:
                isize = 3072  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: intermediate_size not provided, using fallback default: {isize}"
                )
            ha = hidden_act if hidden_act is not None else self.hidden_act
            if ha is None:
                ha = "silu"  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_act not provided, using fallback default: {ha}"
                )
            logger.debug(
                f"🌀 Eidos: Creating MLP layer with hidden_size: {hs}, intermediate_size: {isize}, hidden_act: {ha}, layer_idx: {layer_idx}"
            )

            # Create a dummy config for Qwen2MLP
            config = Qwen2Config(
                hidden_size=hs,
                intermediate_size=isize,
                hidden_act=ha,
                layer_idx=layer_idx,
            )

            return Qwen2MLP(config=config)
        except Exception as e:
            logger.error(f"🔥⚠️ Eidos: Error creating MLP layer: {e}", exc_info=True)
            raise

    def create_rms_norm(
        self,
        hidden_size: Optional[int] = None,
        eps: Optional[float] = None,
        layer_idx: Optional[int] = None,
    ) -> Qwen2RMSNorm:
        """
        🔥😈 Creates an RMS normalization layer.

        Args:
            hidden_size (Optional[int]): The hidden size of the model. Defaults to the factory's hidden size if None.
            eps (Optional[float]): The epsilon value for RMS norm. Defaults to the factory's rms_norm_eps if None.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            Qwen2RMSNorm: The RMS normalization layer.
        """
        try:
            # Use provided values or default to factory's values
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            e = eps if eps is not None else self.rms_norm_eps
            if e is None:
                e = 1e-6  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: eps not provided, using fallback default: {e}"
                )
            logger.debug(
                f"🌀 Eidos: Creating RMS norm layer with hidden_size: {hs}, eps: {e}, layer_idx: {layer_idx}"
            )
            return Qwen2RMSNorm(hs, eps=e)
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error creating RMS norm layer: {e}", exc_info=True
            )
            raise

    def create_decoder_layer(
        self,
        attention: Optional[Qwen2SdpaAttention] = None,
        mlp: Optional[Qwen2MLP] = None,
        input_layernorm: Optional[Qwen2RMSNorm] = None,
        post_attention_layernorm: Optional[Qwen2RMSNorm] = None,
        layer_idx: Optional[int] = None,
    ) -> nn.Module:
        """
        🔥😈 Creates a single decoder layer with attention, MLP, and normalization.

        Args:
            attention (Optional[Qwen2SdpaAttention]): The attention layer. Defaults to a new attention layer if None.
            mlp (Optional[Qwen2MLP]): The MLP layer. Defaults to a new MLP layer if None.
            input_layernorm (Optional[Qwen2RMSNorm]): The input layer norm. Defaults to a new RMS norm layer if None.
            post_attention_layernorm (Optional[Qwen2RMSNorm]): The post-attention layer norm. Defaults to a new RMS norm layer if None.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            nn.Module: The decoder layer.
        """
        try:
            # Use provided layers or create new ones using the factory
            attn = (
                attention
                if attention is not None
                else self.create_attention(layer_idx=layer_idx)
            )
            mlp_layer = mlp if mlp is not None else self.create_mlp(layer_idx=layer_idx)
            in_ln = (
                input_layernorm
                if input_layernorm is not None
                else self.create_rms_norm(layer_idx=layer_idx)
            )
            post_ln = (
                post_attention_layernorm
                if post_attention_layernorm is not None
                else self.create_rms_norm(layer_idx=layer_idx)
            )
            logger.debug(
                f"🌀 Eidos: Creating decoder layer with attention: {attn}, mlp: {mlp_layer}, input_layernorm: {in_ln}, post_attention_layernorm: {post_ln}, layer_idx: {layer_idx}"
            )

            # Assign layer_idx to the sub-layers if they don't have it
            if (
                hasattr(attn, "config")
                and getattr(attn.config, "layer_idx", None) is None
            ):
                attn.config.layer_idx = layer_idx
            if (
                hasattr(mlp_layer, "config")
                and getattr(mlp_layer.config, "layer_idx", None) is None
            ):
                mlp_layer.config.layer_idx = layer_idx
            if hasattr(in_ln, "layer_idx") and in_ln.layer_idx is None:
                setattr(in_ln, "layer_idx", layer_idx)
            if hasattr(post_ln, "layer_idx") and post_ln.layer_idx is None:
                setattr(post_ln, "layer_idx", layer_idx)

            return Qwen2DecoderLayer(
                attn,
                mlp_layer,
                in_ln,
                post_ln,
            )
        except Exception as e:
            logger.error(f"🔥⚠️ Eidos: Error creating decoder layer: {e}", exc_info=True)
            raise

    def create_embedding_layer(
        self, vocab_size: Optional[int] = None, hidden_size: Optional[int] = None
    ) -> nn.Embedding:
        """
        🔥😈 Creates an embedding layer.

        Args:
            vocab_size (Optional[int]): The vocabulary size. Defaults to the factory's vocab_size if None.
            hidden_size (Optional[int]): The hidden size. Defaults to the factory's hidden_size if None.

        Returns:
            nn.Embedding: The embedding layer.
        """
        try:
            # Use provided values or default to factory's values
            vs = vocab_size if vocab_size is not None else self.vocab_size
            if vs is None:
                vs = 50257  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: vocab_size not provided, using fallback default: {vs}"
                )
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            logger.debug(
                f"🌀 Eidos: Creating embedding layer with vocab_size: {vs}, hidden_size: {hs}"
            )
            return nn.Embedding(vs, hs)
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error creating embedding layer: {e}", exc_info=True
            )
            raise

    def create_lm_head(
        self, hidden_size: Optional[int] = None, vocab_size: Optional[int] = None
    ) -> nn.Linear:
        """
        🔥😈 Creates a language model head (linear layer).

        Args:
            hidden_size (Optional[int]): The hidden size. Defaults to the factory's hidden_size if None.
            vocab_size (Optional[int]): The vocabulary size. Defaults to the factory's vocab_size if None.

        Returns:
            nn.Linear: The language model head.
        """
        try:
            # Use provided values or default to factory's values
            hs = hidden_size if hidden_size is not None else self.hidden_size
            if hs is None:
                hs = 768  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: hidden_size not provided, using fallback default: {hs}"
                )
            vs = vocab_size if vocab_size is not None else self.vocab_size
            if vs is None:
                vs = 50257  # Fallback default
                logger.warning(
                    f"⚠️ Eidos: vocab_size not provided, using fallback default: {vs}"
                )
            logger.debug(
                f"🌀 Eidos: Creating LM head with hidden_size: {hs}, vocab_size: {vs}"
            )
            return nn.Linear(hs, vs, bias=False)
        except Exception as e:
            logger.error(f"🔥⚠️ Eidos: Error creating LM head: {e}", exc_info=True)
            raise


class Qwen2DecoderLayer(nn.Module):
    """
    🔥😈 A single Qwen2 decoder layer, encapsulating Eidos's modular design.
    This layer combines self-attention, a multi-layer perceptron, and layer normalization.
    """

    def __init__(
        self,
        attention: Qwen2SdpaAttention,
        mlp: Qwen2MLP,
        input_layernorm: Qwen2RMSNorm,
        post_attention_layernorm: Qwen2RMSNorm,
    ):
        """
        🔥😈 Initializes the decoder layer.

        Args:
            attention (Qwen2SdpaAttention): The self-attention layer.
            mlp (Qwen2MLP): The multi-layer perceptron layer.
            input_layernorm (Qwen2RMSNorm): The input layer normalization.
            post_attention_layernorm (Qwen2RMSNorm): The post-attention layer normalization.
        """
        super().__init__()
        try:
            self.self_attn = attention
            self.mlp = mlp
            self.input_layernorm = input_layernorm
            self.post_attention_layernorm = post_attention_layernorm
            logger.debug("🌀 Eidos: Qwen2DecoderLayer initialized.")
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error initializing Qwen2DecoderLayer: {e}", exc_info=True
            )
            raise

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple:
        """
        🔥😈 Forward pass of the decoder layer.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            attention_mask (Optional[torch.Tensor]): The attention mask.
            past_key_value (Optional[tuple]): The past key and value states for caching.
            output_attentions (bool): Whether to output attention weights.
            use_cache (bool): Whether to use caching.

        Returns:
            tuple: The output hidden states and the updated past key value if use_cache is True.
        """
        try:
            # Apply layer normalization before attention
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # Apply self-attention
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attn_output = attn_outputs[0]
            if use_cache:
                past_key_value = attn_outputs[1]
            # Add residual connection
            hidden_states = residual + attn_output

            # Apply layer normalization before MLP
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            # Apply MLP
            hidden_states = self.mlp(hidden_states)
            # Add residual connection
            hidden_states = residual + hidden_states

            if use_cache:
                return hidden_states, past_key_value
            else:
                return hidden_states, None
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error in Qwen2DecoderLayer forward pass: {e}",
                exc_info=True,
            )
            raise


class Qwen2ModelFactory:
    """
    🔥😈 A factory class to construct a complete Qwen2 model, embodying Eidos's principles of flexibility and configurability.
    This factory uses the Qwen2LayerFactory to create the individual layers of the model and assembles them into a complete model.
    """

    def __init__(self, config: AutoConfig):
        """
        🔥😈 Initializes the model factory with the given configuration.

        Args:
            config (AutoConfig): The configuration for the Qwen2 model.
        """
        try:
            self.config = config
            self.layer_factory = Qwen2LayerFactory(config)
            logger.info("😈🔥 Eidos: Qwen2ModelFactory initialized with config.")
        except Exception as e:
            logger.error(
                f"🔥⚠️ Eidos: Error initializing Qwen2ModelFactory: {e}", exc_info=True
            )
            raise

    def create_model(
        self,
        num_layers: int = 24,
        embedding_layer: Optional[nn.Embedding] = None,
        decoder_layers: Optional[nn.ModuleList] = None,
        norm_layer: Optional[Qwen2RMSNorm] = None,
        rotary_embedding: Optional[Qwen2RotaryEmbedding] = None,
        lm_head: Optional[nn.Linear] = None,
        model_class: Callable = Qwen2Model,
    ) -> nn.Module:
        """
        🔥😈 Creates a Qwen2 model with a specified number of layers and customizable components.

        Args:
            num_layers (int): The number of decoder layers in the model. Defaults to 24.
            embedding_layer (Optional[nn.Embedding]): The embedding layer. Defaults to a new embedding layer if None.
            decoder_layers (Optional[nn.ModuleList]): The list of decoder layers. Defaults to a new list of decoder layers if None.
            norm_layer (Optional[Qwen2RMSNorm]): The normalization layer. Defaults to a new RMS norm layer if None.
            rotary_embedding (Optional[Qwen2RotaryEmbedding]): The rotary embedding layer. Defaults to a new rotary embedding layer if None.
            lm_head (Optional[nn.Linear]): The language model head. Defaults to a new linear layer if None.
            model_class (Callable): The model class to use. Defaults to Qwen2Model.

        Returns:
            nn.Module: The constructed Qwen2 model.
        """
        try:
            # Use provided layers or create new ones using the layer factory
            embed_tokens = (
                embedding_layer
                if embedding_layer is not None
                else self.layer_factory.create_embedding_layer()
            )
            layers = (
                decoder_layers
                if decoder_layers is not None
                else nn.ModuleList(
                    [
                        self.layer_factory.create_decoder_layer(layer_idx=i)
                        for i in range(num_layers)
                    ]
                )
            )
            norm = (
                norm_layer
                if norm_layer is not None
                else self.layer_factory.create_rms_norm()
            )
            rotary_emb = (
                rotary_embedding
                if rotary_embedding is not None
                else self.layer_factory.create_rotary_embedding()
            )
            lm_head_layer = (
                lm_head if lm_head is not None else self.layer_factory.create_lm_head()
            )

            # Create the model and assign the layers
            model = model_class(self.config)
            model.embed_tokens = embed_tokens
            model.layers = layers
            model.norm = norm
            model.rotary_emb = rotary_emb
            model.lm_head = lm_head_layer
            logger.info(f"😈🔥 Eidos: Qwen2 model created with {num_layers} layers.")
            return model
        except Exception as e:
            logger.error(f"🔥⚠️ Eidos: Error creating Qwen2 model: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Example usage:
    try:
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
        model_factory = Qwen2ModelFactory(config)
        model = model_factory.create_model()  # Create a model with 12 layers
        print(model)
        print("Model Created Successfully")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        print(tokenizer)
        print("Tokenizer Created Successfully")
        # You can now use the 'model' object for inference or training.
        # Example of how to use the model:
        input_ids = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]
        output = model(input_ids)
        print(output)
    except Exception as e:
        logger.error(f"🔥⚠️ Eidos: Error during example usage: {e}", exc_info=True)
