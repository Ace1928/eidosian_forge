import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Callable, Dict, Optional, Union


class DynamicActivation(nn.Module):
    """
    A custom PyTorch module that dynamically selects and applies an activation function.
    """

    def __init__(self, activation_func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(DynamicActivation, self).__init__()
        self.activation_func: Callable[[torch.Tensor], torch.Tensor] = activation_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_func(x)


class ActivationDictionary:
    """
    A class designed to encapsulate a comprehensive dictionary of activation functions.
    This class provides a structured approach to accessing various activation functions
    through lambda expressions, facilitating dynamic selection and application within
    neural network architectures.

    Attributes:
        activation_types (Dict[str, DynamicActivation]): A dictionary mapping activation function names to their
                                 corresponding lambda expressions, enabling dynamic invocation.
    """

    def __init__(self) -> None:
        """
        Initializes the ActivationDictionary with a predefined set of activation functions,
        each represented as a lambda expression for dynamic invocation.
        """
        self.activation_types: Dict[str, DynamicActivation] = {
            "ReLU": DynamicActivation(lambda x: F.relu(x)),
            "Sigmoid": DynamicActivation(lambda x: torch.sigmoid(x)),
            "Tanh": DynamicActivation(lambda x: torch.tanh(x)),
            "Softmax": DynamicActivation(
                lambda x: F.softmax(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Linear": DynamicActivation(lambda x: x),
            "ELU": DynamicActivation(lambda x: F.elu(x)),
            "Swish": DynamicActivation(lambda x: x * torch.sigmoid(x)),
            "Leaky ReLU": DynamicActivation(
                lambda x: F.leaky_relu(x, negative_slope=0.01)
            ),
            "Parametric ReLU": DynamicActivation(
                lambda x, a=0.01: F.prelu(
                    x.unsqueeze(0), torch.tensor([a], dtype=torch.float32)
                ).squeeze(0)
            ),
            "ELU-PA": DynamicActivation(lambda x, a=0.01: F.elu(x, alpha=a)),
            "GELU": DynamicActivation(lambda x: F.gelu(x)),
            "Softplus": DynamicActivation(lambda x: F.softplus(x)),
            "Softsign": DynamicActivation(lambda x: F.softsign(x)),
            "Bent Identity": DynamicActivation(
                lambda x: (torch.sqrt(x**2 + 1) - 1) / 2 + x
            ),
            "Hard Sigmoid": DynamicActivation(lambda x: F.hardsigmoid(x)),
            "Mish": DynamicActivation(lambda x: x * torch.tanh(F.softplus(x))),
            "SELU": DynamicActivation(lambda x: F.selu(x)),
            "SiLU": DynamicActivation(lambda x: x * torch.sigmoid(x)),
            "Softshrink": DynamicActivation(lambda x: F.softshrink(x)),
            "Threshold": DynamicActivation(
                lambda x, threshold=0.1, value=0: F.threshold(x, threshold, value)
            ),
            "LogSigmoid": DynamicActivation(lambda x: F.logsigmoid(x)),
            "Hardtanh": DynamicActivation(lambda x: F.hardtanh(x)),
            "ReLU6": DynamicActivation(lambda x: F.relu6(x)),
            "RReLU": DynamicActivation(lambda x: F.rrelu(x)),
            "PReLU": DynamicActivation(
                lambda x, a=0.25: F.prelu(
                    x.unsqueeze(0), torch.tensor([a], dtype=torch.float32)
                ).squeeze(0)
            ),
            "CReLU": DynamicActivation(lambda x: torch.cat((F.relu(x), F.relu(-x)))),
            "ELiSH": DynamicActivation(
                lambda x: torch.sign(x) * (F.elu(torch.abs(x)) + 1) / 2
            ),
            "Hardshrink": DynamicActivation(lambda x: F.hardshrink(x)),
            "LogSoftmax": DynamicActivation(
                lambda x: F.log_softmax(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Softmin": DynamicActivation(
                lambda x: F.softmin(x.unsqueeze(0), dim=0).squeeze(0)
            ),
            "Tanhshrink": DynamicActivation(lambda x: F.tanhshrink(x)),
            "LReLU": DynamicActivation(lambda x: F.leaky_relu(x, negative_slope=0.05)),
            "AReLU": DynamicActivation(lambda x, a=0.1: F.rrelu(x, lower=a, upper=a)),
            "Maxout": DynamicActivation(lambda x: torch.max(x)),
        }

    def get_activation_function(self, name: str) -> Optional[DynamicActivation]:
        """
        Retrieves an activation function by name.

        Parameters:
            name (str): The name of the activation function to retrieve.

        Returns:
            Optional[DynamicActivation]: The activation function as a lambda expression.
        """
        return self.activation_types.get(name, None)
