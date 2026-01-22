# Definition of activation functions using PyTorch operations, encapsulated in lambda expressions
        self.activation_types = {
            "ReLU": lambda x: torch.relu(torch.tensor(x, dtype=torch.complex64)).item(),
            "Sigmoid": lambda x: torch.sigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Tanh": lambda x: torch.tanh(torch.tensor(x, dtype=torch.complex64)).item(),
            "Softmax": lambda x: torch.softmax(
                torch.tensor([x], dtype=torch.complex64), dim=0
            ).tolist(),
            "Linear": lambda x: x,
            "ELU": lambda x: torch.nn.functional.elu(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Swish": lambda x: x
            * torch.sigmoid(torch.tensor(x, dtype=torch.complex64)).item(),
            "Leaky ReLU": lambda x: torch.nn.functional.leaky_relu(
                torch.tensor(x, dtype=torch.complex64), negative_slope=0.01
            ).item(),
            "Parametric ReLU": lambda x, a=0.01: torch.nn.functional.prelu(
                torch.tensor([x], dtype=torch.complex64), torch.tensor([a])
            ).item(),
            "ELU-PA": lambda x, a=0.01: torch.nn.functional.elu(
                torch.tensor(x, dtype=torch.complex64), alpha=a
            ).item(),
            "GELU": lambda x: torch.nn.functional.gelu(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Softplus": lambda x: torch.nn.functional.softplus(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Softsign": lambda x: torch.nn.functional.softsign(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
            "Bent Identity": lambda x: (
                (torch.sqrt(torch.tensor(x, dtype=torch.complex64) ** 2 + 1) - 1) / 2
                + x
            ).item(),
            "Hard Sigmoid": lambda x: torch.nn.functional.hardsigmoid(
                torch.tensor(x, dtype=torch.complex64)
            ).item(),
        }
