import numpy as np
import math
import logging
from Algorithm import Algorithm
from Constants import USER_SEED

# Setting up logging configuration with maximum verbosity and detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Seed the numpy random number generator for reproducibility
np.random.seed(USER_SEED)


def sigmoid(m: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid activation function to the input matrix.

    Parameters:
        m (np.ndarray): The input matrix to which the sigmoid function will be applied.

    Returns:
        np.ndarray: The result of applying the sigmoid function to the input matrix.

    Raises:
        RuntimeError: If an error occurs during the function application.
    """
    logging.debug(f"Applying sigmoid activation function to the input: {m}")
    try:
        result = 1 / (1 + np.exp(-m))
        logging.debug(f"Sigmoid function output: {result}")
        return result
    except Exception as e:
        logging.error(
            f"An error occurred while applying sigmoid function: {e}", exc_info=True
        )
        raise RuntimeError(f"Sigmoid function application failed due to: {e}") from e


def ReLU(x: np.ndarray) -> np.ndarray:
    """
    Apply the Rectified Linear Unit (ReLU) activation function to the input matrix.

    Parameters:
        x (np.ndarray): The input matrix to which the ReLU function will be applied.

    Returns:
        np.ndarray: The result of applying the ReLU function to the input matrix.

    Raises:
        RuntimeError: If an error occurs during the function application.
    """
    logging.debug(f"Applying ReLU activation function to the input: {x}")
    try:
        result = x * (x > 0)
        logging.debug(f"ReLU function output: {result}")
        return result
    except Exception as e:
        logging.error(
            f"An error occurred while applying ReLU function: {e}", exc_info=True
        )
        raise RuntimeError(f"ReLU function application failed due to: {e}") from e


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Apply the hyperbolic tangent activation function to the input matrix.

    Parameters:
        x (np.ndarray): The input matrix to which the tanh function will be applied.

    Returns:
        np.ndarray: The result of applying the tanh function to the input matrix.

    Raises:
        RuntimeError: If an error occurs during the function application.
    """
    logging.debug(f"Applying tanh activation function to the input: {x}")
    try:
        result = np.tanh(x)
        logging.debug(f"tanh function output: {result}")
        return result
    except Exception as e:
        logging.error(
            f"An error occurred while applying tanh function: {e}", exc_info=True
        )
        raise RuntimeError(f"tanh function application failed due to: {e}") from e


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.1,
    ) -> None:
        """
        Initialize the Neural Network with specified number of input, hidden, and output nodes.

        Parameters:
            input_nodes (int): Number of input nodes.
            hidden_nodes (int): Number of hidden nodes.
            output_nodes (int): Number of output nodes.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        ...
        self.shape = (input_nodes, hidden_nodes, output_nodes)
        logging.info(f"Initialized NeuralNetwork with shape {self.shape}")
        self.initialize()

    def initialize(self):
        """
        Initialize the weights and biases of the network randomly.
        """
        self.biases = [
            np.random.randn(i) for i in [self.hidden_nodes, self.output_nodes]
        ]
        self.weights = [
            np.random.randn(j, i)
            for i, j in zip(
                [self.input_nodes, self.hidden_nodes],
                [self.hidden_nodes, self.output_nodes],
            )
        ]
        logging.debug(f"Network weights initialized: {self.weights}")
        logging.debug(f"Network biases initialized: {self.biases}")

    def feedforward(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Perform a feedforward pass through the network using the input matrix.

        Parameters:
            input_matrix (np.ndarray): The input matrix for the feedforward pass.

        Returns:
            np.ndarray: The output matrix after the feedforward pass.

        Raises:
            RuntimeError: If an error occurs during the feedforward pass.
        """
        input_matrix = np.array(input_matrix)
        logging.debug(f"Starting feedforward pass with input: {input_matrix}")
        try:
            for b, w in zip(self.biases, self.weights):
                input_matrix = tanh(np.dot(w, input_matrix) + b)
                logging.debug(f"Feedforward pass, layer output: {input_matrix}")
        except Exception as e:
            logging.error(
                f"An error occurred during the feedforward pass: {e}", exc_info=True
            )
            raise RuntimeError(f"Feedforward pass failed due to: {e}") from e
        return input_matrix

    def crossover(self, networkA: "NeuralNetwork", networkB: "NeuralNetwork"):
        """
        Perform crossover between two networks to create a new network.

        Parameters:
            networkA (NeuralNetwork): The first parent network.
            networkB (NeuralNetwork): The second parent network.
        """
        weightsA = networkA.weights.copy()
        weightsB = networkB.weights.copy()

        biasesA = networkA.biases.copy()
        biasesB = networkB.biases.copy()

        for i in range(len(self.weights)):
            length = len(self.weights[i])
            split = np.random.randint(1, length)
            self.weights[i] = weightsA[i].copy()
            self.weights[i][split:] = weightsB[i][split:].copy()

        for i in range(len(self.biases)):
            length = len(self.biases[i])
            split = np.random.randint(1, length)
            self.biases[i] = biasesA[i].copy()
            self.biases[i][:split] = biasesB[i][:split].copy()
        logging.info("Crossover operation completed.")

    def mutation(self, a: float, val: float) -> float:
        """
        Mutate a weight or bias with a certain probability.

        Parameters:
            a (float): The original value of the weight or bias.
            val (float): The mutation probability.

        Returns:
            float: The mutated value if mutation occurs, otherwise the original value.
        """
        if np.random.rand() < val:
            mutated_value = np.random.randn()
            logging.debug(f"Mutating value {a} to {mutated_value}")
            return mutated_value
        return a

    def mutate(self, val: float):
        """
        Apply mutation to all weights and biases in the network based on a given probability.

        Parameters:
            val (float): The mutation probability.
        """
        mutation_function = np.vectorize(self.mutation)
        for i in range(len(self.weights)):
            self.weights[i] = mutation_function(self.weights[i], val)
        for i in range(len(self.biases)):
            self.biases[i] = mutation_function(self.biases[i], val)
        logging.info("Mutation operation completed.")

    def print(self):
        """
        Print the current state of the network, including its shape, weights, and biases.
        """
        print("shape", self.shape)
        print("weights", self.weights)
        print("biases", self.biases)

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """
        Predict the output for a given input matrix using the neural network.

        Parameters:
            input_matrix (np.ndarray): The input matrix for prediction.

        Returns:
            np.ndarray: The predicted output matrix.
        """
        return self.feedforward(input_matrix)
