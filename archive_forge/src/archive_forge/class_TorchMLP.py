from typing import Callable, List, Optional, Union, Tuple
from ray.rllib.core.models.torch.utils import Stride2D
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
class TorchMLP(nn.Module):
    """A multi-layer perceptron with N dense layers.

    All layers (except for an optional additional extra output layer) share the same
    activation function, bias setup (use bias or not), and LayerNorm setup
    (use layer normalization or not).

    If `output_dim` (int) is not None, an additional, extra output dense layer is added,
    which might have its own activation function (e.g. "linear"). However, the output
    layer does NOT use layer normalization.
    """

    def __init__(self, *, input_dim: int, hidden_layer_dims: List[int], hidden_layer_activation: Union[str, Callable]='relu', hidden_layer_use_bias: bool=True, hidden_layer_use_layernorm: bool=False, output_dim: Optional[int]=None, output_use_bias: bool=True, output_activation: Union[str, Callable]='linear'):
        """Initialize a TorchMLP object.

        Args:
            input_dim: The input dimension of the network. Must not be None.
            hidden_layer_dims: The sizes of the hidden layers. If an empty list, only a
                single layer will be built of size `output_dim`.
            hidden_layer_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each hidden layer's output and its activation.
            hidden_layer_use_bias: Whether to use bias on all dense layers (excluding
                the possible separate output layer).
            hidden_layer_activation: The activation function to use after each layer
                (except for the output). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
            output_dim: The output dimension of the network. If None, no specific output
                layer will be added and the last layer in the stack will have
                size=`hidden_layer_dims[-1]`.
            output_use_bias: Whether to use bias on the separate output layer,
                if any.
            output_activation: The activation function to use for the output layer
                (if any). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
        """
        super().__init__()
        assert input_dim > 0
        self.input_dim = input_dim
        hidden_activation = get_activation_fn(hidden_layer_activation, framework='torch')
        layers = []
        dims = [self.input_dim] + list(hidden_layer_dims) + ([output_dim] if output_dim else [])
        for i in range(0, len(dims) - 1):
            is_output_layer = output_dim is not None and i == len(dims) - 2
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=output_use_bias if is_output_layer else hidden_layer_use_bias))
            if not is_output_layer:
                if hidden_layer_use_layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                if hidden_activation is not None:
                    layers.append(hidden_activation())
        output_activation = get_activation_fn(output_activation, framework='torch')
        if output_dim is not None and output_activation is not None:
            layers.append(output_activation())
        self.mlp = nn.Sequential(*layers)
        self.expected_input_dtype = torch.float32

    def forward(self, x):
        return self.mlp(x.type(self.expected_input_dtype))