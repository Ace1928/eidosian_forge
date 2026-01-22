import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
from typing import List, Tuple
import types
import importlib.util
import logging


__all__ = [
    "import_from_path",
    "StandardDecorator",
    "setup_logging",
    "dynamic_depth_expectimax",
    "adjust_depth_based_on_complexity",
    "expectimax",
    "heuristic_evaluation",
    "simulate_move",
    "get_empty_tiles",
    "is_game_over",
    "calculate_best_move",
    "short_term_memory",
    "lru_memory",
    "short_to_long_term_memory_transfer",
    "long_term_memory_optimisation",
    "long_term_memory_learning",
    "long_term_memory_pruning",
    "long_term_memory_retrieval",
    "long_term_memory_update",
    "long_term_memory_storage",
    "long_term_memory_indexing",
    "long_term_memory_backup",
    "long_term_memory_restore",
    "long_term_memory_clear",
    "long_term_memory_search",
    "long_term_memory_retrieval",
    "long_term_memory_update",
    "short_term_memory_update",
    "short_term_memory_storage",
    "short_term_memory_clear",
    "short_term_memory_search",
    "short_term_memory_retrieval",
    "short_term_memory_backup",
    "short_term_memory_restore",
    "short_term_memory_to_long_term_memory_transfer",
    "long_term_memory_to_short_term_memory_transfer",
    "optimise_game_strategy",
    "learn_from_game_data",
    "analyse_game_data",
    "visualise_game_data",
    "search_game_data",
    "retrieve_game_data",
    "update_game_data",
    "store_game_data",
    "clear_game_data",
    "export_game_data",
    "import_game_data",
    "genetic_algorithm",
    "reinforcement_learning",
    "deep_q_learning",
    "monte_carlo_tree_search",
    "alpha_beta_pruning",
    "minimax_algorithm",
    "expectimax_algorithm",
    "evaluate_game",
    "optimise_game",
    "predict_game_outcomes",
    "analyse_game_state",
    "search_game_state",
    "retrieve_game_state",
    "update_game_state",
    "store_game_state",
    "clear_game_state",
    "export_game_state",
    "import_game_state",
    "analyse_prediction_accuracy",
    "analyse_previous_moves",
    "analyse_prediction_errors",
    "anlayse_learning_progress",
    "introspective_analysis",
    "optimise_learning_rate",
    "alter_learning_strategy",
    "adapt_decision_making",
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
    Performs the expectimax search on a given board state to a specified depth.

    This function recursively explores all possible moves in the game tree, alternating between the player's turn and chance nodes,
    to evaluate the board's heuristic value. It aims to find the optimal move for the player by maximizing the score while considering
    the probability of different tiles appearing in chance nodes.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search.
        playerTurn (bool): Flag indicating whether it's the player's turn or a chance node.

    Returns:
        Tuple[float, str]: The best heuristic value found and the corresponding move.
    """
    logging.debug(f"Starting expectimax with depth {depth} and playerTurn {playerTurn}")
    if depth == 0 or is_game_over(board):
        heuristic = heuristic_evaluation(board)
        logging.info(f"Terminal node or game over reached with heuristic {heuristic}")
        return heuristic, ""

    if playerTurn:
        best_value = float("-inf")
        best_move = ""
        for move in ["up", "down", "left", "right"]:
            new_board, _ = simulate_move(board, move)
            value, _ = expectimax(new_board, depth - 1, False)
            if value > best_value:
                best_value = value
                best_move = move
            logging.debug(f"Player move {move} evaluated with value {value}")
        logging.info(f"Best move for player: {best_move} with value {best_value}")
        return best_value, best_move
    else:
        total_value = 0
        empty_tiles = get_empty_tiles(board)
        num_empty = len(empty_tiles)
        if num_empty == 0:
            heuristic = heuristic_evaluation(board)
            logging.info(f"No empty tiles, heuristic evaluation: {heuristic}")
            return heuristic, ""
        for i, j in empty_tiles:
            for val in [2, 4]:
                new_board = np.array(board)
                new_board[i, j] = val
                value, _ = expectimax(new_board, depth - 1, True)
                probability = 0.9 if val == 2 else 0.1
                total_value += value * probability / num_empty
                logging.debug(
                    f"Chance node with tile {val} at ({i},{j}) evaluated with value {value}"
                )
        logging.info(f"Total value for chance node: {total_value}")
        return total_value, ""


@StandardDecorator()
def heuristic_evaluation(board: np.ndarray) -> float:
    """
    Evaluates the heuristic value of the board for the 2048 game.

    Args:
        board (np.ndarray): The game board.

    Returns:
        float: The heuristic value of the board.
    """
    empty_tiles = len(get_empty_tiles(board))
    max_tile = np.max(board)
    smoothness, monotonicity = calculate_smoothness_and_monotonicity(board)

    # Calculate smoothness and monotonicity
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
        (empty_tiles * 2.7) + (np.log2(max_tile) * 0.9) + smoothness + monotonicity
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


@StandardDecorator()
def short_term_memory(board: np.ndarray, move: str, score: int) -> None:
    """
    Stores the current game state, move, and score in short-term memory for future reference.

    Args:
        board (np.ndarray): The current game board.
        move (str): The move made.
        score (int): The score gained from the move.
    """
    pass


@StandardDecorator()
def lru_memory(board: np.ndarray, move: str, score: int) -> None:
    """
    Implements a Least Recently Used (LRU) memory cache to store game states, moves, and scores.

    Args:
        board (np.ndarray): The current game board.
        move (str): The move made.
        score (int): The score gained from the move.
    """
    pass


@StandardDecorator()
def short_to_long_term_memory_transfer() -> None:
    """
    Transfers relevant information from short-term memory to long-term memory for learning and optimisation.
    """
    pass


# Advanced AI Strategy using RandomForest for dynamic learning
class AdvancedAIStrategy:

    @StandardDecorator()
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.training_data = []
        self.target_scores = []

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


# Example usage
# ai_strategy = AdvancedAIStrategy()
# During gameplay, continuously update training data and retrain model
# ai_strategy.update_training_data(board_state, score)
# ai_strategy.train_model()
# Use the model to predict scores and make decisions
# predicted_score = ai_strategy.predict_score(next_board_state)

# Continued in ai_logic.py

# Integration of Deep Learning for decision making


class DeepLearningDecisionMaker:

    @StandardDecorator()
    def __init__(self):
        self.model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(16,)),
                Dense(64, activation="relu"),
                Dense(4, activation="softmax"),  # Assuming 4 possible moves
            ]
        )
        self.model.compile(optimizer="adam", loss="categorical_crossentropy")

    def train_model(self, game_data, move_labels):
        self.model.fit(game_data, move_labels, epochs=10)

    @StandardDecorator()
    def predict_move(self, board_state):
        return self.model.predict(board_state.flatten().reshape(1, -1))

    """
    Makes the final decision based on the game policy and learning outcomes.
    """
    pass


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
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def predict(self, x):
        hidden_layer = self.relu(np.dot(x, self.weights1))
        output_layer = self.softmax(np.dot(hidden_layer, self.weights2))
        return output_layer


class DeepLearningDecisionMaker:

    @StandardDecorator()
    def __init__(self):
        self.model = SimplePerceptron(
            16, 64, 4
        )  # Assuming a 4x4 board and 4 possible moves

    @StandardDecorator()
    def train_model(self, game_data, move_labels):
        # Placeholder for training logic
        pass

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
