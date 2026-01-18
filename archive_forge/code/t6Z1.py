import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
import types
import importlib.util
import logging
import collections
from typing import Deque, Dict, Tuple, List


__all__ = [
    "import_from_path",
    "StandardDecorator",
    "setup_logging",
    "dynamic_depth_expectimax",
    "adjust_depth_based_on_complexity",
    "expectimax",
    "heuristic_evaluation",
    "get_empty_tiles",
    "is_game_over",
    "calculate_best_move",
    "ShortTermMemory",
    "LRUMemory",
    "short_to_long_term_memory_transfer",
    "DynamicLearningStrategy",
    "SimplePerceptron",
    "DeepLearningDecisionMaker",
]


def import_from_path(name: str, path: str) -> types.ModuleType:
    """
    Dynamically imports a module from a given file path.

    Args:
        name (str): The name of the module.
        path (str): The file path to the module.

    Returns:
        types.ModuleType: The imported module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


standard_decorator = import_from_path(
    "standard_decorator", "/home/lloyd/EVIE/standard_decorator.py"
)
StandardDecorator = standard_decorator.StandardDecorator
setup_logging = standard_decorator.setup_logging

setup_logging()


@StandardDecorator()
def dynamic_depth_expectimax(
    board: np.ndarray, playerTurn: bool, initial_depth: int = 3
) -> Tuple[float, str]:
    """
    Calculates the best move using the expectimax algorithm with dynamic depth adjustment based on the game state complexity.

    Args:
        board (np.ndarray): The current game board.
        playerTurn (bool): Flag indicating whether it's the player's turn or a chance node.
        initial_depth (int): The initial depth for the search, adjusted dynamically.

    Returns:
        Tuple[float, str]: The best heuristic value found and the corresponding move.
    """
    empty_tiles = len(get_empty_tiles(board))
    depth = adjust_depth_based_on_complexity(initial_depth, empty_tiles)
    return expectimax(board, depth, playerTurn)


@StandardDecorator()
def adjust_depth_based_on_complexity(initial_depth: int, empty_tiles: int) -> int:
    """
    Adjusts the search depth based on the complexity of the game state, represented by the number of empty tiles.

    Args:
        initial_depth (int): The initial search depth.
        empty_tiles (int): The number of empty tiles on the board.

    Returns:
        int: The adjusted depth.
    """
    if empty_tiles > 10:
        return max(2, initial_depth - 1)  # Less complex, shallower search
    elif empty_tiles < 4:
        return initial_depth + 1  # More complex, deeper search
    else:
        return initial_depth


@StandardDecorator()
def expectimax(board: np.ndarray, depth: int, playerTurn: bool) -> Tuple[float, str]:
    """
    Performs the expectimax search to evaluate the best move for the current game state by exploring all possible moves
    and their outcomes up to a specified depth. This function alternates between maximizing the player's score and
    evaluating the expected value of chance nodes, thereby simulating the game's stochastic nature.

    Args:
        board (np.ndarray): The current game board represented as a 2D NumPy array.
        depth (int): The depth of the search, indicating how many moves ahead the algorithm should evaluate.
        playerTurn (bool): A boolean flag indicating whether it's the player's turn to move or a chance node.

    Returns:
        Tuple[float, str]: A tuple containing the best heuristic value found for the current player's turn and the
                            corresponding move as a string. If it's a chance node, the move will be an empty string.
    """
    # Log the start of the expectimax function with the current depth and player turn.
    logging.debug(
        f"Initiating expectimax with depth {depth} and playerTurn {playerTurn}"
    )

    # Base case: if the search has reached its maximum depth or the game is over, evaluate the board.
    if depth == 0 or is_game_over(board):
        heuristic = heuristic_evaluation(board)
        logging.info(f"Reached base case with heuristic value: {heuristic}")
        return heuristic, ""

    # Player's turn: aim to maximize the score by choosing the best possible move.
    if playerTurn:
        best_value = float("-inf")  # Initialize the best value as negative infinity.
        best_move = ""  # Initialize the best move as an empty string.
        # Iterate over all possible moves.
        for move in ["left", "right", "up", "down"]:
            # Simulate the move and get the resulting board state.
            new_board, _, _ = calculate_best_move(board, move)
            # Recursively call expectimax for the next depth, switching to the chance node.
            value, _ = expectimax(new_board, depth - 1, False)
            # Update the best value and move if the current value is greater.
            if value > best_value:
                best_value = value
                best_move = move
            # Log the evaluated move and its value.
            logging.debug(f"Evaluating player move: {move} with value: {value}")
        # Log the best move found for the player.
        logging.info(f"Best move for player: {best_move} with value: {best_value}")
        return best_value, best_move
    # Chance node's turn: calculate the expected value of all possible tile placements.
    else:
        total_value = 0  # Initialize the total value for the chance node.
        empty_tiles = get_empty_tiles(board)  # Get all empty tiles on the board.
        # Iterate over all empty tiles and possible tile values (2 and 4).
        for i, j in empty_tiles:
            for value in [2, 4]:
                new_board = np.array(board)  # Create a copy of the board.
                new_board[i, j] = value  # Place the tile on the board.
                # Recursively call expectimax for the next depth, switching to the player's turn.
                value, _ = expectimax(new_board, depth - 1, True)
                # Calculate the probability of the tile value appearing.
                probability = 0.9 if value == 2 else 0.1
                # Update the total value based on the probability and the evaluated value.
                total_value += probability * value / len(empty_tiles)
                # Log the evaluation of the chance node.
                logging.debug(
                    f"Chance node with tile {value} at ({i},{j}) evaluated with value: {value}"
                )
        # Log the total value calculated for the chance node.
        logging.info(f"Total value for chance node: {total_value}")
        return total_value, ""


@StandardDecorator()
def heuristic_evaluation(board: np.ndarray) -> float:
    """
    Performs an advanced heuristic evaluation of the game board by integrating multiple factors
    such as tile arrangement in a snake pattern, the presence of empty tiles, the value of the
    highest tile, its smoothness, and monotonicity. It further incorporates penalties for non-optimal
    placements of the highest tile, enhancing the decision-making process for the AI.

    This method meticulously calculates the heuristic value by considering the strategic importance
    of each factor, ensuring a robust and comprehensive evaluation that guides the AI towards
    making informed decisions that maximize its chances of winning.

    Args:
        board (np.ndarray): The current state of the game board, represented as a 2D NumPy array.

    Returns:
        float: A calculated heuristic value representing the evaluated state of the board, factoring
               in all the strategic elements considered critical for the game's success.
    """
    # Establishing a snake pattern for strategic heuristic evaluation. This pattern aids in
    # assessing the board's current state by prioritizing tile arrangement in a specific order
    # that mimics a snake-like traversal, which is crucial for maximizing score potential.
    snake_pattern = np.array(
        [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]], dtype=int
    )
    # The snake pattern is designed to guide the AI in maintaining a preferred tile arrangement,
    # encouraging the accumulation of higher-value tiles in a manner that supports subsequent
    # high-value merges, thereby optimizing the board's scoring potential.
    flat_board = board.flatten()
    snake_scores = np.zeros_like(flat_board)
    for i, val in enumerate(flat_board):
        snake_scores[snake_pattern.flatten()[i]] = val

    # Encourage monotonic decrease along the snake
    snake_score = np.sum(snake_scores / 10 ** np.arange(16))

    # Penalize if the highest tile is not in a corner
    max_tile_penalty = 0
    if np.argmax(flat_board) not in [0, 3, 12, 15]:
        max_tile = np.max(flat_board)
        max_tile_penalty = np.sqrt(max_tile)

    # Calculate additional heuristic components
    empty_tiles = len(get_empty_tiles(board))
    max_tile = np.max(board)
    smoothness, monotonicity = calculate_smoothness_and_monotonicity(board)

    # Calculate smoothness and monotonicity
    @StandardDecorator()
    def calculate_smoothness_and_monotonicity(board: np.ndarray) -> Tuple[float, float]:
        smoothness = 0
        monotonicity_up_down = 0
        monotonicity_left_right = 0

        # Calculate smoothness
        for i in range(board.shape[0]):
            for j in range(board.shape[1] - 1):
                if board[i, j] != 0 and board[i, j + 1] != 0:
                    smoothness -= abs(np.log2(board[i, j]) - np.log2(board[i, j + 1]))
                if board[j, i] != 0 and board[j + 1, i] != 0:
                    smoothness -= abs(np.log2(board[j, i]) - np.log2(board[j + 1, i]))

        # Calculate monotonicity
        for i in range(board.shape[0]):
            for j in range(1, board.shape[1]):
                if board[i, j - 1] > board[i, j]:
                    monotonicity_left_right += np.log2(board[i, j - 1]) - np.log2(
                        board[i, j]
                    )
                else:
                    monotonicity_left_right -= np.log2(board[i, j]) - np.log2(
                        board[i, j - 1]
                    )

                if board[j - 1, i] > board[j, i]:
                    monotonicity_up_down += np.log2(board[j - 1, i]) - np.log2(
                        board[j, i]
                    )
                else:
                    monotonicity_up_down -= np.log2(board[j, i]) - np.log2(
                        board[j - 1, i]
                    )

        return smoothness, (monotonicity_left_right + monotonicity_up_down) / 2

    heuristic_value = (
        (empty_tiles * 2.7)
        + (np.log2(max_tile) * 0.9)  # Weight for the highest tile
        + smoothness  # Smoothness of the tile arrangement
        + monotonicity  # Monotonicity in tile values
        + snake_score  # Score based on snake pattern
        - max_tile_penalty  # Penalty for misplaced highest tile
    )
    return heuristic_value


@StandardDecorator()
def get_empty_tiles(board: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds all empty tiles on the board.

    Args:
        board (np.ndarray): The game board.

    Returns:
        List[Tuple[int, int]]: A list of coordinates for the empty tiles.
    """
    return list(zip(*np.where(board == 0)))


@StandardDecorator()
def is_game_over(board: np.ndarray) -> bool:
    """
    Checks if the game is over (no moves left).

    Args:
        board (np.ndarray): The game board.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    return not any(
        simulate_move(board, move)[0].tolist() != board.tolist()
        for move in ["up", "down", "left", "right"]
    )


@StandardDecorator()
def calculate_best_move(board: np.ndarray) -> str:
    _, best_move = dynamic_depth_expectimax(board, playerTurn=True, initial_depth=3)
    return best_move


# Implementing a class to manage short-term memory for the AI
class ShortTermMemory:
    """
    Manages the short-term memory for the AI, storing recent moves and their outcomes.
    """

    def __init__(self, capacity: int = 10):
        self.memory: Deque[Tuple[np.ndarray, str, int]] = collections.deque(
            maxlen=capacity
        )

    @StandardDecorator
    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the short-term memory.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        self.memory.append((board, move, score))


# Implementing a class to manage LRU memory for the AI
class LRUMemory:
    """
    Implements a Least Recently Used (LRU) memory cache to store game states, moves, and scores.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.cache: Dict[Tuple[str, int], np.ndarray] = collections.OrderedDict()

    @StandardDecorator
    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the LRU memory, evicting the least recently used item if necessary.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        key = (move, score)
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = board
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Implementing a function to transfer data from short-term to long-term memory
@StandardDecorator()
def short_to_long_term_memory_transfer(
    short_term_memory: ShortTermMemory, long_term_memory: LRUMemory
) -> None:
    """
    Transfers relevant information from short-term memory to long-term memory for learning and optimisation.

    Args:
        short_term_memory (ShortTermMemory): The short-term memory instance.
        long_term_memory (LRUMemory): The long-term memory instance.
    """
    while short_term_memory.memory:
        board, move, score = short_term_memory.memory.popleft()
        long_term_memory.store(board, move, score)


# Example usage
short_term_memory = ShortTermMemory()
lru_memory = LRUMemory()
# Assuming board, move, and score are obtained from the game
# short_term_memory.store(board, move, score)
# short_to_long_term_memory_transfer(short_term_memory, lru_memory)


class DynamicLearningStrategy:

    @StandardDecorator()
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.training_data = []
        self.target_scores = []

    @StandardDecorator()
    def update_training_data(self, board_state, score):
        # Flatten the board state and append it to the training data
        self.training_data.append(board_state.flatten())
        self.target_scores.append(score)

    @StandardDecorator()
    def train_model(self):
        # Train the model once sufficient data is collected
        if len(self.training_data) > 100:
            self.model.fit(self.training_data, self.target_scores)

    @StandardDecorator()
    def predict_score(self, board_state):
        # Predict the score for a given board state
        return self.model.predict([board_state.flatten()])[0]


class SimplePerceptron:

    @StandardDecorator()
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    @StandardDecorator()
    def relu(self, x):
        return np.maximum(0, x)

    @StandardDecorator()
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    @StandardDecorator()
    def predict(self, x):
        hidden_layer = self.relu(np.dot(x, self.weights1))
        output_layer = self.softmax(np.dot(hidden_layer, self.weights2))
        return output_layer


class NeuralNetwork:

    @StandardDecorator()
    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        """
        Initializes a simple neural network with one hidden layer.

        Args:
            input_size (int): The size of the input layer.
            hidden_layers (list): A list containing the size of each hidden layer.
            output_size (int): The size of the output layer.
        """
        self.weights = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            )

    @StandardDecorator()
    def relu(self, x):
        return np.maximum(0, x)

    @StandardDecorator()
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @StandardDecorator()
    def predict(self, x):
        """
        Predicts the next move based on the current board state.

        Args:
            x (np.ndarray): The current state of the game board, flattened.

        Returns:
            np.ndarray: The probabilities of each move.
        """
        for i in range(len(self.weights) - 1):
            x = self.relu(np.dot(x, self.weights[i]))
        return self.softmax(np.dot(x, self.weights[-1]))


class DeepLearningDecisionMaker:
    """
    A decision maker that uses a deep learning model to predict the best move in the game.

    """

    @StandardDecorator()
    def __init__(self):
        self.model = SimplePerceptron(
            16, 64, 4
        )  # Assuming a 4x4 board and 4 possible moves
        self.model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(16,)),
                Dense(64, activation="relu"),
                Dense(4, activation="softmax"),  # Assuming 4 possible moves
            ]
        )

        self.model.compile(optimizer="adam", loss="categorical_crossentropy")
        # self.model = RandomForestRegressor(n_estimators=100) #Base Python Random Forest Regressor To Be Developed


# Example instantiation and usage
nn = NeuralNetwork(16, [64, 64], 4)
board_state = np.random.rand(1, 16)  # Example board state
predicted_moves = nn.predict(board_state)
print("Predicted move probabilities:", predicted_moves)


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        self.weights = [
            np.random.randn(prev, next) * 0.1
            for prev, next in zip(
                [input_size] + hidden_layers, hidden_layers + [output_size]
            )
        ]
        self.biases = [np.zeros((1, next)) for next in hidden_layers + [output_size]]

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, x):
        for w, b in zip(
            self.weights[:-1], self.biases[:-1]
        ):  # Apply ReLU to all layers except the last
            x = self.relu(np.dot(x, w) + b)
        return self.softmax(
            np.dot(x, self.weights[-1]) + self.biases[-1]
        )  # Apply softmax to the output layer


class DeepLearningDecisionMaker:
    @StandardDecorator()
    def __init__(self):
        self.model = NeuralNetwork(16, [64, 64], 4)  # Use the custom NeuralNetwork
        self.training_data = []
        self.target_scores = []

    @StandardDecorator()
    def predict_move(self, board_state: np.ndarray) -> int:
        prediction = self.model.predict(board_state.flatten().reshape(1, -1))
        return np.argmax(
            prediction
        )  # Returns the index of the highest probability move

    @StandardDecorator()
    def train_network(
        model: NeuralNetwork,
        training_data: np.ndarray,
        target_scores: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
    ):
        """
        Trains the neural network model using a comprehensive training cycle.

        Args:
            model (NeuralNetwork): The neural network model to train.
            training_data (np.ndarray): The input data for training.
            target_scores (np.ndarray): The target scores for each input.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the gradient descent.
        """
        for epoch in range(epochs):
            # Forward propagation
            predictions = model.predict(training_data)

            # Loss calculation (Mean Squared Error)
            loss = np.mean((predictions - target_scores) ** 2)

            # Backpropagation (simplified)
            # Placeholder for backpropagation logic to adjust weights and biases
            # In a real scenario, this would involve calculating gradients and adjusting weights

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    @StandardDecorator()
    def backpropagation(
        model: NeuralNetwork,
        input_data: np.ndarray,
        target: np.ndarray,
        learning_rate: float,
    ):
        """
        Performs a simplified backpropagation algorithm to adjust the weights and biases of the neural network.

        Args:
            model (NeuralNetwork): The neural network model.
            input_data (np.ndarray): The input data used for training.
            target (np.ndarray): The target output.
            learning_rate (float): The learning rate for adjustments.
        """
        # Forward pass to get output predictions and intermediate activations
        activations = [input_data]
        x = input_data
        for w, b in zip(model.weights[:-1], model.biases[:-1]):
            x = model.relu(np.dot(x, w) + b)
            activations.append(x)
        output = model.softmax(np.dot(x, model.weights[-1]) + model.biases[-1])
        activations.append(output)

        # Calculate error derivative for the output layer
        error = output - target

        # Backward pass to propagate the error and update weights and biases
        for i in reversed(range(len(model.weights))):
            activation = activations[i]
            if i == len(model.weights) - 1:  # Output layer
                delta = error
            else:  # Hidden layers
                delta = np.dot(delta, model.weights[i + 1].T) * (activation > 0).astype(
                    float
                )  # Derivative of ReLU

            weight_gradient = np.dot(activations[i - 1].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)

            # Update weights and biases
            model.weights[i] -= learning_rate * weight_gradient
            model.biases[i] -= learning_rate * bias_gradient

    @ StandardDecorator() @ StandardDecorator()
    def update_training_data(self, board_state, score):
        self.training_data.append(board_state.flatten())
        self.target_scores.append(score)

    @StandardDecorator()
    def train_model(self):
        if len(self.training_data) > 100:  # Start training after collecting enough data
            self.model.fit(self.training_data, self.target_scores)

    @StandardDecorator()
    def predict_score(self, board_state):
        return self.model.predict([board_state.flatten()])[0]

    @StandardDecorator()
    def predict_move(self, board_state):
        prediction = self.model.predict(board_state.flatten().reshape(1, -1))
        return np.argmax(
            prediction
        )  # Simplified to return the index of the highest probability move

    @StandardDecorator()
    def simulate_move(board: np.ndarray, move: str) -> Tuple[np.ndarray, int]:
        """
        Simulates a move on the board and returns the new board state and score gained.

        This function shifts the tiles in the specified direction and combines tiles of the same value.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move to simulate ('up', 'down', 'left', 'right').

        Returns:
            Tuple[np.ndarray, int]: The new board state and score gained from the move.
        """

        @StandardDecorator()
        def shift_and_combine(row: list) -> Tuple[list, int]:
            """
            Shifts non-zero elements to the left and combines elements of the same value.
            Args:
                row (list): A row (or column) from the game board.
            Returns:
                Tuple[list, int]: The shifted and combined row, and the score gained.
            """
            non_zero = [i for i in row if i != 0]  # Filter out zeros
            combined = []
            score = 0
            skip = False
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    combined.append(2 * non_zero[i])
                    score += 2 * non_zero[i]
                    skip = True
                else:
                    combined.append(non_zero[i])
            combined.extend(
                [0] * (len(row) - len(combined))
            )  # Fill the rest with zeros
            return combined, score

        @StandardDecorator()
        def rotate_board(board: np.ndarray, move: str) -> np.ndarray:
            """
            Rotates the board to simplify shifting logic.
            Args:
                board (np.ndarray): The game board.
                move (str): The move direction.
            Returns:
                np.ndarray: The rotated board.
            """
            if move == "up":
                return board.T
            elif move == "down":
                return np.rot90(board, 2).T
            elif move == "left":
                return board
            elif move == "right":
                return np.rot90(board, 2)
            else:
                raise ValueError("Invalid move direction")

        rotated_board = rotate_board(board, move)
        new_board = np.zeros_like(board)
        total_score = 0
        for i, row in enumerate(rotated_board):
            new_row, score = shift_and_combine(list(row))
            total_score += score
            new_board[i] = new_row

        if move in ["up", "down"]:
            new_board = new_board.T
        elif move == "right":
            new_board = np.rot90(new_board, 2)

        return new_board, total_score
