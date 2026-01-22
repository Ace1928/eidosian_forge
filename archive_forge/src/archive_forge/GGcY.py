import kan


class KAN:
    def __init__(
        self, input_size: int, output_size: int, depth: int, activation: str = "tanh"
    ):
        """
        Initialize the Kolmogorov-Arnold Network (KAN).

        Args:
            input_size (int): Number of input neurons.
            output_size (int): Number of output neurons.
            depth (int): Number of layers in the network.
            activation (str): Activation function to use ('tanh', 'relu', 'sigmoid').
        """
        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.activation = self._get_activation_function(activation)
        self.weights = [
            np.random.randn(input_size, output_size).astype(np.float32)
            for _ in range(depth)
        ]
        self.biases = [
            np.random.randn(output_size).astype(np.float32) for _ in range(depth)
        ]

    def _get_activation_function(self, name: str) -> Callable:
        """
        Get the activation function based on the name.

        Args:
            name (str): Name of the activation function.

        Returns:
            Callable: Activation function.
        """
        activations = {
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        }
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after passing through the network.
        """
        for i in range(self.depth):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = self.activation(x)
        return x

    def mutate(self, mutation_rate: float = 0.1):
        """
        Mutate the network's weights and biases.

        Args:
            mutation_rate (float): Probability of mutation for each weight and bias.
        """
        for i in range(self.depth):
            if random.random() < mutation_rate:
                self.weights[i] += (
                    np.random.randn(*self.weights[i].shape).astype(np.float32)
                    * mutation_rate
                )
                self.biases[i] += (
                    np.random.randn(*self.biases[i].shape).astype(np.float32)
                    * mutation_rate
                )

    def inherit(self, other: "KAN"):
        """
        Inherit weights and biases from another KAN instance.

        Args:
            other (KAN): Another KAN instance to inherit from.
        """
        new_depth = max(
            1,
            self.depth
            + (1 if random.random() < 0.05 else -1 if random.random() < 0.05 else 0),
        )
        new_depth = min(new_depth, len(self.weights), len(other.weights))
        self.weights = [
            (self.weights[i] + other.weights[i]) / 2 for i in range(new_depth)
        ]
        self.biases = [(self.biases[i] + other.biases[i]) / 2 for i in range(new_depth)]
        self.depth = new_depth

    def save(self, filename: str):
        """
        Save the network's weights and biases to a file.

        Args:
            filename (str): The file path to save the network.
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {"weights": self.weights, "biases": self.biases, "depth": self.depth}, f
            )

    @classmethod
    def load(cls, filename: str) -> "KAN":
        """
        Load a network's weights and biases from a file.

        Args:
            filename (str): The file path to load the network from.

        Returns:
            KAN: An instance of the KAN class with loaded weights and biases.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            input_size=data["weights"][0].shape[0],
            output_size=data["biases"][0].shape[0],
            depth=data["depth"],
        )
        instance.weights = data["weights"]
        instance.biases = data["biases"]
        return instance

    def train(
        self,
        dataset: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
    ):
        """
        Train the network using a simple gradient descent algorithm.

        Args:
            dataset (np.ndarray): Input data for training.
            targets (np.ndarray): Target outputs for training.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            for x, y in zip(dataset, targets):
                output = self.forward(x)
                error = y - output
                self._backpropagate(x, error, learning_rate)

    def _backpropagate(self, x: np.ndarray, error: np.ndarray, learning_rate: float):
        """
        Backpropagate the error and update the weights and biases.

        Args:
            x (np.ndarray): Input data.
            error (np.ndarray): Error between predicted and actual output.
            learning_rate (float): Learning rate for gradient descent.
        """
        for i in reversed(range(self.depth)):
            delta = error * self._activation_derivative(self.activation, x)
            self.weights[i] += learning_rate * np.outer(x, delta)
            self.biases[i] += learning_rate * delta
            error = np.dot(delta, self.weights[i].T)

    def _activation_derivative(self, activation: Callable, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Args:
            activation (Callable): Activation function.
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Derivative of the activation function.
        """
        if activation == np.tanh:
            return 1 - np.tanh(x) ** 2
        elif activation == np.maximum:
            return np.where(x > 0, 1, 0)
        elif activation == (lambda x: 1 / (1 + np.exp(-x))):
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError(
                "Unsupported activation function for derivative calculation."
            )
