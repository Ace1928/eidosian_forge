import torch
import torch.nn.functional as F


class ActivationDictionary:
    """
    A class designed to encapsulate a comprehensive dictionary of activation functions.
    This class provides a structured approach to accessing various activation functions
    through lambda expressions, facilitating dynamic selection and application within
    neural network architectures.

    Attributes:
        activation_types (dict): A dictionary mapping activation function names to their
                                 corresponding lambda expressions, enabling dynamic invocation.
    """

    def __init__(self):
        """
        Initializes the ActivationDictionary with a predefined set of activation functions,
        each represented as a lambda expression for dynamic invocation.
        """
        self.activation_types = {
            "ReLU": lambda x: F.relu(torch.tensor(x, dtype=torch.complex64)).item(),
            "Sigmoid": lambda x: torch.sigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Tanh": lambda x: torch.tanh(torch.tensor(x, dtype=torch.complex64)).item(),
            "Softmax": lambda x: F.softmax(
                torch.tensor([x], dtype=torch.complex64), dim=0
            ).tolist(),
            "Linear": lambda x: x,
            "ELU": lambda x: F.elu(torch.tensor(x, dtype=torch.complex64)).item(),
            "Swish": lambda x: x
            * torch.sigmoid(torch.tensor(x, dtype=torch.complex64)).item(),
            "Leaky ReLU": lambda x: F.leaky_relu(
                torch.tensor(x, dtype=torch.complex64), negative_slope=0.01
            ).item(),
            "Parametric ReLU": lambda x, a=0.01: F.prelu(
                torch.tensor([x], dtype=torch.complex64), torch.tensor([a])
            ).item(),
            "ELU-PA": lambda x, a=0.01: F.elu(
                torch.tensor(x, dtype=torch.complex64), alpha=a
            ).item(),
            "GELU": lambda x: F.gelu(torch.tensor(x, dtype=torch.complex64)).item(),
            "Softplus": lambda x: F.softplus(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Softsign": lambda x: F.softsign(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Bent Identity": lambda x: (
                (torch.sqrt(torch.tensor(x, dtype=torch.complex64) ** 2 + 1) - 1) / 2
                + x
            ).item(),
            "Hard Sigmoid": lambda x: F.hardsigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Mish": lambda x: x
            * torch.tanh(F.softplus(torch.tensor(x, dtype=torch.complex64))).item(),
            "SELU": lambda x: F.selu(torch.tensor(x, dtype=torch.complex64)).item(),
            "SiLU": lambda x: x
            * torch.sigmoid(torch.tensor(x, dtype=torch.complex64)).item(),
            "Softshrink": lambda x: F.softshrink(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Threshold": lambda x, threshold=0.1, value=0: F.threshold(
                torch.tensor(x, dtype=torch.complex64), threshold, value
            ).item(),
            "LogSigmoid": lambda x: F.logsigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Hardtanh": lambda x: F.hardtanh(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "ReLU6": lambda x: F.relu6(torch.tensor(x, dtype=torch.complex64)).item(),
            "RReLU": lambda x: F.rrelu(torch.tensor(x, dtype=torch.complex64)).item(),
            "PReLU": lambda x, a=0.25: F.prelu(
                torch.tensor([x], dtype=torch.complex64), torch.tensor([a])
            ).item(),
            "CReLU": lambda x: torch.cat(
                (
                    F.relu(torch.tensor(x, dtype=torch.complex64)),
                    F.relu(-torch.tensor(x, dtype=torch.complex64)),
                )
            ).item(),
            "ELiSH": lambda x: (
                torch.sign(torch.tensor(x, dtype=torch.complex64))
                * (F.elu(abs(torch.tensor(x, dtype=torch.complex64))) + 1)
                / 2
            ).item(),
            "Hardshrink": lambda x: F.hardshrink(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "LogSoftmax": lambda x: F.log_softmax(
                torch.tensor([x], dtype=torch.complex64), dim=0
            ).tolist(),
            "Softmin": lambda x: F.softmin(
                torch.tensor([x], dtype=torch.complex64), dim=0
            ).tolist(),
            "Tanhshrink": lambda x: F.tanhshrink(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "LReLU": lambda x: F.leaky_relu(
                torch.tensor(x, dtype=torch.complex64), negative_slope=0.05
            ).item(),
            "AReLU": lambda x, a=0.1: F.rrelu(
                torch.tensor(x, dtype=torch.complex64), lower=a, upper=a
            ).item(),
            "Maxout": lambda x: torch.max(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
        }

    def get_activation_function(self, name):
        """
        Retrieves an activation function by name.

        Parameters:
            name (str): The name of the activation function to retrieve.

        Returns:
            function: The activation function as a lambda expression.
        """
        return self.activation_types.get(name, None)
