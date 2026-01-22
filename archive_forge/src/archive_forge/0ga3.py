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
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not load module {name} from path {path}")


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
    logging.info(
        f"Starting dynamic depth expectimax with initial depth {initial_depth}."
    )
    empty_tiles = len(get_empty_tiles(board))
    depth = adjust_depth_based_on_complexity(initial_depth, empty_tiles)
    logging.info(f"Adjusted depth based on complexity: {depth}.")
    best_heuristic_value, best_move = expectimax(board, depth, playerTurn)
    logging.info(
        f"Best heuristic value: {best_heuristic_value}, Best move: {best_move}."
    )
    return best_heuristic_value, best_move


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
    logging.debug(f"Adjusting depth based on {empty_tiles} empty tiles.")
    if empty_tiles > 10:
        adjusted_depth = max(2, initial_depth - 1)  # Less complex, shallower search
    elif empty_tiles < 4:
        adjusted_depth = initial_depth + 1  # More complex, deeper search
    else:
        adjusted_depth = initial_depth
    logging.debug(f"Depth adjusted to {adjusted_depth}.")
    return adjusted_depth


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
    logging.debug(
        f"Initiating expectimax with depth {depth} and playerTurn {playerTurn}."
    )
    if depth == 0 or is_game_over(board):
        heuristic = heuristic_evaluation(board)
        logging.info(f"Reached base case with heuristic value: {heuristic}.")
        return heuristic, ""
    if playerTurn:
        best_value, best_move = float("-inf"), ""
        for move in ["left", "right", "up", "down"]:
            new_board, _, _ = calculate_best_move(board, move)
            value, _ = expectimax(new_board, depth - 1, False)
            if value > best_value:
                best_value, best_move = value, move
            logging.debug(f"Evaluating player move: {move} with value: {value}.")
        logging.info(f"Best move for player: {best_move} with value: {best_value}.")
        return best_value, best_move
    else:
        total_value, empty_tiles = 0, get_empty_tiles(board)
        for i, j in empty_tiles:
            for value in [2, 4]:
                new_board = np.array(board)
                new_board[i, j] = value
                value, _ = expectimax(new_board, depth - 1, True)
                probability = 0.9 if value == 2 else 0.1
                total_value += probability * value / len(empty_tiles)
                logging.debug(
                    f"Chance node with tile {value} at ({i},{j}) evaluated with value: {value}."
                )
        logging.info(f"Total value for chance node: {total_value}.")
        return total_value, ""


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
                monotonicity_up_down += np.log2(board[j - 1, i]) - np.log2(board[j, i])
            else:
                monotonicity_up_down -= np.log2(board[j, i]) - np.log2(board[j - 1, i])

    return smoothness, (monotonicity_left_right + monotonicity_up_down) / 2


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
    logging.debug("Starting heuristic evaluation.")
    snake_pattern = np.array(
        [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]], dtype=int
    )
    flat_board = board.flatten()
    snake_scores = np.zeros_like(flat_board)
    for i, val in enumerate(flat_board):
        snake_scores[snake_pattern.flatten()[i]] = val
    snake_score = np.sum(snake_scores / 10 ** np.arange(16))
    max_tile_penalty = 0
    if np.argmax(flat_board) not in [0, 3, 12, 15]:
        max_tile = np.max(flat_board)
        max_tile_penalty = np.sqrt(max_tile)
    empty_tiles = len(get_empty_tiles(board))
    max_tile = np.max(board)
    smoothness, monotonicity = calculate_smoothness_and_monotonicity(board)
    heuristic_value = (
        (empty_tiles * 2.7)
        + (np.log2(max_tile) * 0.9)
        + smoothness
        + monotonicity
        + snake_score
        - max_tile_penalty
    )
    logging.debug(f"Calculated heuristic value: {heuristic_value}.")
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
    logging.debug("Identifying empty tiles on the board.")
    empty_tiles = list(zip(*np.where(board == 0)))
    logging.debug(f"Found {len(empty_tiles)} empty tiles.")
    return empty_tiles


@StandardDecorator()
def is_game_over(board: np.ndarray) -> bool:
    """
    Checks if the game is over (no moves left).

    Args:
        board (np.ndarray): The game board.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    logging.debug("Checking if the game is over.")
    game_over = not any(
        DeepLearningDecisionMaker.simulate_move(board, move)[0].tolist()
        != board.tolist()
        for move in ["up", "down", "left", "right"]
    )
    logging.debug(f"Game over status: {game_over}.")
    return game_over


@StandardDecorator()
def calculate_best_move(board: np.ndarray) -> str:
    """
    Determines the best move for the current board state using the dynamic depth expectimax algorithm.

    Args:
        board (np.ndarray): The current game board.

    Returns:
        str: The best move determined.
    """
    logging.debug("Calculating the best move for the current board state.")
    _, best_move = dynamic_depth_expectimax(board, playerTurn=True, initial_depth=3)
    logging.info(f"Best move calculated: {best_move}.")
    return best_move


# Implementing a class to manage short-term memory for the AI. This memory stores recent moves and their outcomes.
class ShortTermMemory:
    """
    Manages the short-term memory for the AI, storing recent moves and their outcomes.
    """

    @StandardDecorator()
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


# Implementing a class to manage LRU memory for the AI. Acts as a ranked working memory for game states, moves, and scores.
class LRUMemory:
    """
    Implements a Least Recently Used (LRU) memory cache to store game states, moves, and scores.
    """

    @StandardDecorator()
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


# Implementing a class to manage the learning strategy for the AI. Acts as a long-term memory for game states, moves, and scores.
class LongTermMemory:
    """
    Manages the long-term memory for the AI, storing relevant game states, moves, and scores for learning and optimisation.
    Utilises reversible encoding/decoding, compression and vectorization of all stored long term values.
    Using pickle for serialization and deserialization of the string representation of the vectorized data after all encoding and compression.
    Deserialised and then decompressed and then decoded back to original string from the deserialised decompressed devectorized value.
    Implementing efficient indexing and retrieval of stored data for training and decision-making.
    Utilising a ranking system to determine the most relevant data for decision-making.
    Utilsing a mechanism to normalise and standardise data stored to long term memory to ensure no duplication or redundancy.
    Using a hashing mechanism to ensure data integrity and consistency and uniqueness of stored data.
    """

    @StandardDecorator()
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory: Dict[str, Tuple[np.ndarray, str, int]] = {}

    @StandardDecorator()
    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the long-term memory.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        key = self._hash(board)
        if key not in self.memory:
            self.memory[key] = (board, move, score)
            if len(self.memory) > self.capacity:
                self._remove_least_relevant()

    @StandardDecorator()
    def _remove_least_relevant(self) -> None:
        """
        Removes the least relevant item from the long-term memory based on a ranking system.
        """
        # Placeholder for the removal logic based on a ranking system
        pass

    @StandardDecorator()
    def retrieve(self, board: np.ndarray) -> Tuple[np.ndarray, str, int]:
        """
        Retrieves the stored move and score for the given game board from the long-term memory.

        Args:
            board (np.ndarray): The game board.

        Returns:
            Tuple[np.ndarray, str, int]: The stored board, move, and score.
        """
        key = self._hash(board)
        return self.memory.get(key, (None, None, None))

    @StandardDecorator()
    def _encode(self, board: np.ndarray) -> str:
        """
        Encodes the game board into a string representation for storage.

        Args:
            board (np.ndarray): The game board.

        Returns:
            str: The encoded string representation of the board.
        """
        # Placeholder for encoding logic
        return board.tostring()

    @StandardDecorator()
    def _decode(self, encoded_board: str) -> np.ndarray:
        """
        Decodes the encoded string representation of the board back into a NumPy array.

        Args:
            encoded_board (str): The encoded string representation of the board.

        Returns:
            np.ndarray: The decoded game board.
        """
        # Placeholder for decoding logic
        return np.frombuffer(encoded_board)

    @StandardDecorator()
    def _compress(self, data: str) -> str:
        """
        Compresses the given data to reduce memory usage.

        Args:
            data (str): The data to compress.

        Returns:
            str: The compressed data.
        """
        # Placeholder for compression logic
        return data

    @StandardDecorator()
    def _decompress(self, compressed_data: str) -> str:
        """
        Decompresses the compressed data back to its original form.

        Args:
            compressed_data (str): The compressed data.

        Returns:
            str: The decompressed data.
        """
        # Placeholder for decompression logic
        return compressed_data

    @StandardDecorator()
    def _vectorize(self, data: str) -> np.ndarray:
        """
        Vectorizes the data for efficient storage and retrieval.

        Args:
            data (str): The data to vectorize.

        Returns:
            np.ndarray: The vectorized data.
        """
        # Placeholder for vectorization logic
        return np.array([ord(char) for char in data])

    @StandardDecorator()
    def _devectorize(self, vectorized_data: np.ndarray) -> str:
        """
        Devectorizes the vectorized data back to its original form.

        Args:
            vectorized_data (np.ndarray): The vectorized data.

        Returns:
            str: The devectorized data.
        """
        # Placeholder for devectorization logic
        return "".join([chr(int(val)) for val in vectorized_data])

    @StandardDecorator()
    def _serialize(self, data: str) -> str:
        """
        Serializes the data for storage.

        Args:
            data (str): The data to serialize.

        Returns:
            str: The serialized data.
        """
        # Placeholder for serialization logic
        return data

    @StandardDecorator()
    def _deserialize(self, serialized_data: str) -> str:
        """
        Deserializes the serialized data back to its original form.

        Args:
            serialized_data (str): The serialized data.

        Returns:
            str: The deserialized data.
        """
        # Placeholder for deserialization logic
        return serialized_data

    @StandardDecorator()
    def _normalize(self, data: str) -> str:
        """
        Normalizes the data to ensure consistency and uniqueness.

        Args:
            data (str): The data to normalize.

        Returns:
            str: The normalized data.
        """
        # Placeholder for normalization logic
        return data

    @StandardDecorator()
    def _standardize(self, data: str) -> str:
        """
        Standardizes the data to ensure no duplication or redundancy.

        Args:
            data (str): The data to standardize.

        Returns:
            str: The standardized data.
        """
        # Placeholder for standardization logic
        return data

    @StandardDecorator()
    def _hash(self, board: np.ndarray) -> str:
        """
        Generates a hash key for the given game board.

        Args:
            board (np.ndarray): The game board.

        Returns:
            str: The hashed key for the board.
        """
        return str(hash(board.tostring()))

    @StandardDecorator()
    def _unhash(self, key: str) -> np.ndarray:
        """
        Retrieves the game board from the hashed key.

        Args:
            key (str): The hashed key for the board.

        Returns:
            np.ndarray: The game board.
        """
        return np.frombuffer(key)

    @StandardDecorator()
    def _rank(self, data: str) -> int:
        """
        Ranks the data based on relevance for decision-making.

        Args:
            data (str): The data to rank.

        Returns:
            int: The ranking value.
        """
        # Placeholder for ranking logic
        return 0

    @StandardDecorator()
    def _rank_all(self) -> None:
        """
        Ranks all data in the memory based on relevance.
        """
        for key in self.memory:
            self.memory[key] = (
                self.memory[key][0],
                self.memory[key][1],
                self.memory[key][2],
                self._rank(key),
            )  # Update the ranking value
            ranked_memory = sorted(
                self.memory.items(), key=lambda x: x[1][3], reverse=True
            )  # Sort the memory based on the ranking value
            self.memory = dict(
                ranked_memory
            )  # Update the memory with the sorted values
        return None

    @StandardDecorator()
    def _update(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Updates the memory with the given game state, move, and score.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.
        """
        key = self._hash(board)
        if key in self.memory:
            self.memory[key] = (board, move, score)
        else:
            self.store(board, move, score)


# Implementing a function to transfer data from short-term to long-term memory
@StandardDecorator()
def short_to_LRU_memory_transfer(
    short_term_memory: ShortTermMemory, long_term_memory: LRUMemory
) -> None:
    """
    Transfers relevant information from short-term memory to long-term memory for learning and optimisation.

    Args:
        short_term_memory (ShortTermMemory): The short-term memory instance.
        LRU_memory (LRUMemory): The long-term memory instance.
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


@StandardDecorator()
def LRU_to_long_term_memory_transfer(
    LRU_memory: LRUMemory, long_term_memory: LongTermMemory
) -> None:
    """
    Transfers relevant information from LRU memory to long-term memory for learning and optimisation.

    Args:
        LRU_memory (LRUMemory): The LRU memory instance.
        long_term_memory (LongTermMemory): The long-term memory instance.
    """
    for key, value in LRU_memory.cache.items():
        board, move, score = value
        long_term_memory.store(board, move, score)


# Example usage
lru_memory = LRUMemory()
long_term_memory = LongTermMemory()
# Assuming board, move, and score are obtained from the game
# lru_memory.store(board, move, score)
# LRU_to_long_term_memory_transfer(lru_memory, long_term_memory)


@StandardDecorator()
def long_term_memory_to_LRU_transfer(
    long_term_memory: LongTermMemory, LRU_memory: LRUMemory
) -> None:
    """
    Transfers relevant information from long-term memory to LRU memory for efficient decision-making.

    Args:
        long_term_memory (LongTermMemory): The long-term memory instance.
        LRU_memory (LRUMemory): The LRU memory instance.
    """
    for key, value in long_term_memory.memory.items():
        board, move, score = value
        LRU_memory.store(board, move, score)
        del long_term_memory.memory[key]


# Example usage
long_term_memory = LongTermMemory()
lru_memory = LRUMemory()
# Assuming board, move, and score are obtained from the game
# long_term_memory.store(board, move, score)
# long_term_memory_to_LRU_transfer(long_term_memory, lru_memory)


@StandardDecorator()
def long_term_memory_to_short_term_transfer(
    long_term_memory: LongTermMemory, short_term_memory: ShortTermMemory
) -> None:
    """
    Transfers relevant information from long-term memory to short-term memory for immediate decision-making.

    Args:
        long_term_memory (LongTermMemory): The long-term memory instance.
        short_term_memory (ShortTermMemory): The short-term memory instance.
    """
    for key, value in long_term_memory.memory.items():
        board, move, score = value
        short_term_memory.store(board, move, score)
        del long_term_memory.memory[key]


# Example usage
long_term_memory = LongTermMemory()
short_term_memory = ShortTermMemory()
# Assuming board, move, and score are obtained from the game
# long_term_memory.store(board, move, score)
# long_term_memory_to_short_term_transfer(long_term_memory, short_term_memory)


@StandardDecorator
def dynamic_learning_strategy(
    LRU_Memory: LRUMemory,
    Short_Term_Memory: ShortTermMemory,
    Long_Term_Memory: LongTermMemory,
    PPO: ProximalPolicyOptimization,
    GA: GeneticAlgorithm,
    Training_Data: np.ndarray,
    Target_Scores: np.ndarray,
    Neural_Network: DeepLearningDecisionMaker,
) -> np.ndarray:
    """
    Implements a dynamic learning strategy that transfers data between short-term, LRU, and long-term memory based on the current game state.
    And also uses a Proximal Policy Optimization algorithm augmented by a genetic algorithm to optimize the neural network weights.

    Args:
        LRU_Memory (LRUMemory): The LRU memory instance.
        Short_Term_Memory (ShortTermMemory): The short-term memory instance.
        Long_Term_Memory (LongTermMemory): The long-term memory instance.
        PPO (Proximal Policy Optimization): The PPO algorithm instance.
        GA (Genetic Algorithm): The Genetic Algorithm instance.
        Training Data (np.ndarray): The training data for the neural network.
        Target Scores (np.ndarray): The target scores for training the neural network.
        Neural Network (DeepLearningDecisionMaker): The neural network instance.

    Returns:
        Prediction (np.ndarray): The predicted output from the neural network corresponding to the current game state.
    """

    # Transfer data between short-term, LRU, and long-term memory based on the current game state
    if Neural_Network.predict(Training_Data) == Target_Scores:
        LRU_to_long_term_memory_transfer(LRU_Memory, Long_Term_Memory)
        long_term_memory_to_short_term_transfer(Long_Term_Memory, Short_Term_Memory)
    else:
        short_to_LRU_memory_transfer(Short_Term_Memory, LRU_Memory)
        LRU_to_long_term_memory_transfer(LRU_Memory, Long_Term_Memory)

    # PPO Prediction
    PPO.optimize_weights(Training_Data, Target_Scores)

    # GA Prediction
    GA.optimize_weights(Training_Data, Target_Scores)

    # Combined Prediction
    GA_vs_PPO = GA.optimize_weights(
        Training_Data, Target_Scores
    ) + PPO.optimize_weights(Training_Data, Target_Scores)

    # Average the predictions
    GA_vs_PPO = GA_vs_PPO / 2

    # Train the neural network with the updated data
    Neural_Network.train_network(Training_Data, Target_Scores)

    # Predict the output based on the current game state
    Prediction = Neural_Network.predict(Training_Data)

    # Perform a complex analysis of the 4 different predictions and their target scores and predictions to determine the best prediction
    # based on the current game state and the neural network weights
    # This involves a more complex decision making algorithm such as the cutting edge decision making algorithm
    # that can determine the best prediction based on the current game state and the neural network weights
    # Utilising cutting edge research and modern algorithms to derive and develop and implement a novel efficient asynchronous decision making algorithm
    # inspired by swarm intelligence and deep reinforcement learning and hive mind/group think asynchronous decision making algorithms
    # The general structure of this advanced swarm intelligence decision making algorithm is as follows:

    # 1. Initialize the swarm with random positions and velocities corresponding to the predictions

    # 2. Evaluate the fitness of each prediction based on the target scores and the predictions

    # 3. Update the position and velocity of each prediction based on the fitness evaluation

    # 4. Repeat the evaluation and update process for a specified number of iterations

    # 5. Select the prediction with the highest fitness value as the best prediction

    # The swarm intelligence decision making algorithm is designed to efficiently explore the solution space and converge to the best prediction
    # based on the target scores and the predictions.
    # It is expected to be advantageous over existing algorithms because of its ability to leverage the collective intelligence of the swarm
    # It leverages this utilising algorithmic logic that follows the step by step granular steps as follows:

    # 1. Set up a lognormal randomised distribution of swarm members, each belonging to a "lineage (the swarm is built upon genetic/biologically inspired elements that can mutate)"
    # Each lineage corresponding to a unique prediction from the neural network, PPO, GA and GAvsPPO.

    # 2 Establish each randomised population group for each lineage with randomised attributes that correspond to partial derivatives of the neural network weights, the target scores and the predictions essentially so that the components add up to or multiply or in some way combine to form the predictions and weights (how they combine is lineage dependent)
    # Each lineage has a unique way of combining the components to form the predictions and weights. These unique recombination approaches are determined by paramters that are set up when initially randomised and these parameters are parametrically tied to all other genetic parameters of the lineage.

    # 3. Set up each member with random velocities and positions and delays and weights and biases and activations and learning rates and optimisers and loss functions and activation functions and other hyperparameters that are unique to each lineage and member of the swarm.

    # 4. Set up each connection between each of the swarm members with an oscillating small bias to the connection strength (+ and - some randomised normalised value) that is unique to each member and lineage.

    # 5 Set up each connection between each member with a threshold and a delay value that is unique to each connection between each member and betwen each lineage.

    # 6. Set up each member with a unique set of activation functions and loss functions and optimisers and learning rates and other hyperparameters that are unique to each member and lineage.

    # 7. Set up each member with a unique set of weights and biases that are unique to each member and lineage.

    # 8. Implement a group reinforcement mechanism (neurons that wire together fire togehter and neurons that fire together wire together) that is unique to each member and lineage but as members fire with each other and if the response is positive their responses align more.

    # 9. Implement a group punishment mechanism (neurons that wire together fire together and neurons that fire together wire together) that is unique to each member and lineage but as members fire with each other and if the response is negative their responses align less. Implement this via a method of neurotoxicity so that as this punishment accumulates over time the individual member (acting like a complex multifacated neuron) becomes less response both in speed of response and strength of response

    # 10. Implement a group reward mechanism (neurons that wire together fire together and neurons that fire together wire together) that is unique to each member and lineage but as members fire with each other and if the response is positive their responses align more. Implement this via a method of neuroplasticity so that as this reward accumulates over time the individual member (acting like a complex multifacated neuron) becomes more response both in speed of response and strength of response

    # 11. Implement an individual reward mechanism

    # 12. Implement a cooperation reward mechanism to discourage selfishness and encourage cooperation

    # 13. Implement a collaboration mechanism that encourages sharing positive moves and changes and states

    # 14 Evaluate the fitness of each prediction based on the target scores and the predictions

    # 15 Update the position and velocity of each prediction based on the fitness evaluation

    # 16 Repeat the evaluation and update process for a specified number of iterations

    # 17 Select the prediction with the highest fitness value as the best prediction

    # The swarm intelligence decision making algorithm is designed to efficiently explore the solution space and converge to the best prediction
    # based on the target scores and the predictions.

    # Finally Compare all of the predictions and select the best prediction based on the target scores and the predictions
    # Ensure that all aspects are asynchronous and instead of using floats utilise a programmatically mapped integer range (negative and positive) that floats are normalised to to keep resource usage low but still keep the approximate accuracy of the floats.
    # You could make this more efficient overall by implementing an advanced mapping mechanism where a float with, for example, 8 points of precision was mapped to an integer that was 9 numbers long, with an additional number representing an uncertainty/stochastic value that was used to determine the final value of the float. This would allow for a more efficient use of memory and processing power while still maintaining the accuracy of the float.

    return Prediction


from typing import List, Tuple
from functools import wraps
import logging

# Setting up logging for detailed insights into the neural network operations
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def StandardDecorator():
    """
    A decorator to standardize and log the execution of methods within the neural network classes.
    This decorator aims to provide insights into method calls, arguments passed, and the time taken for execution.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(
                f"Executing {func.__name__} with args: {args} and kwargs: {kwargs}"
            )
            result = func(*args, **kwargs)
            logging.info(f"Executed {func.__name__} successfully.")
            return result

        return wrapper

    return decorator


class NeuralNetwork:
    """
    A class representing a simple neural network with customizable layers, weights, and biases.
    This neural network supports forward propagation with ReLU and softmax activation functions.
    """

    @StandardDecorator()
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int):
        """
        Initializes the NeuralNetwork with random weights and zero biases based on the specified architecture.

        Args:
            input_size (int): The size of the input layer.
            hidden_layers (List[int]): A list of sizes for each hidden layer.
            output_size (int): The size of the output layer.
        """
        self.weights = [
            np.random.randn(prev, next) * 0.1
            for prev, next in zip(
                [input_size] + hidden_layers, hidden_layers + [output_size]
            )
        ]
        self.biases = [np.zeros((1, next)) for next in hidden_layers + [output_size]]
        logging.debug(
            f"NeuralNetwork initialized with weights: {self.weights} and biases: {self.biases}"
        )

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function on the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array with ReLU applied.
        """
        return np.maximum(0, x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function on the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array with softmax applied.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @StandardDecorator()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the neural network using the input data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output from the neural network.
        """
        for w, b in zip(
            self.weights[:-1], self.biases[:-1]
        ):  # Apply ReLU to all layers except the last
            x = self.relu(np.dot(x, w) + b)
        return self.softmax(
            np.dot(x, self.weights[-1]) + self.biases[-1]
        )  # Apply softmax to the output layer


class DeepLearningDecisionMaker:
    """
    A class that encapsulates the neural network and provides methods for training, predicting, and simulating moves.
    """

    @StandardDecorator()
    def __init__(self):
        """
        Initializes the DeepLearningDecisionMaker with a custom NeuralNetwork and empty training data.
        """
        self.model = NeuralNetwork(16, [64, 64], 4)  # Use the custom NeuralNetwork
        self.training_data = []
        self.target_scores = []
        logging.debug(
            "DeepLearningDecisionMaker initialized with an empty model and training data."
        )

    @StandardDecorator()
    def train_network(
        self,
        training_data: np.ndarray,
        target_scores: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
    ):
        """
        Trains the neural network model using the provided training data and target scores over a specified number of epochs.

        Args:
            training_data (np.ndarray): The input data for training.
            target_scores (np.ndarray): The target scores for each input.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the gradient descent.
        """
        for epoch in range(epochs):
            # Forward propagation
            predictions = self.model.predict(training_data)

            # Loss calculation (Mean Squared Error)
            loss = np.mean((predictions - target_scores) ** 2)

            # Placeholder for backpropagation logic to adjust weights and biases
            # In a real scenario, this would involve calculating gradients and adjusting weights

            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    @StandardDecorator()
    def backpropagation(
        self, input_data: np.ndarray, target: np.ndarray, learning_rate: float
    ):
        """
        Performs a simplified backpropagation algorithm to adjust the weights and biases of the neural network.

        Args:
            input_data (np.ndarray): The input data used for training.
            target (np.ndarray): The target output.
            learning_rate (float): The learning rate for adjustments.
        """
        # Forward pass to get output predictions and intermediate activations
        activations = [input_data]
        x = input_data
        for w, b in zip(self.model.weights[:-1], self.model.biases[:-1]):
            x = self.model.relu(np.dot(x, w) + b)
            activations.append(x)
        output = self.model.softmax(
            np.dot(x, self.model.weights[-1]) + self.model.biases[-1]
        )
        activations.append(output)

        # Calculate error derivative for the output layer
        error = output - target

        # Backward pass to propagate the error and update weights and biases
        for i in reversed(range(len(self.model.weights))):
            activation = activations[i]
            if i == len(self.model.weights) - 1:  # Output layer
                delta = error
            else:  # Hidden layers
                delta = np.dot(delta, self.model.weights[i + 1].T) * (
                    activation > 0
                ).astype(
                    float
                )  # Derivative of ReLU

            weight_gradient = np.dot(activations[i - 1].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)

            # Update weights and biases
            self.model.weights[i] -= learning_rate * weight_gradient
            self.model.biases[i] -= learning_rate * bias_gradient

    @StandardDecorator()
    def update_training_data(self, board_state: np.ndarray, score: int):
        """
        Updates the training data with the given board state and score.

        Args:
            board_state (np.ndarray): The current game board state.
            score (int): The score associated with the board state.
        """
        self.training_data.append(board_state.flatten())
        self.target_scores.append(score)

    @StandardDecorator()
    def train_model(self):
        """
        Trains the model using the collected training data if there is enough data collected.
        """
        if len(self.training_data) > 100:  # Start training after collecting enough data
            training_data_np = np.array(self.training_data)
            target_scores_np = np.array(self.target_scores)
            self.train_network(training_data_np, target_scores_np)

    @StandardDecorator()
    def predict_score(self, board_state: np.ndarray) -> float:
        """
        Predicts the score for a given board state using the neural network model.

        Args:
            board_state (np.ndarray): The current game board state.

        Returns:
            float: The predicted score.
        """
        prediction = self.model.predict([board_state.flatten()])[0]
        return prediction

    @StandardDecorator()
    def predict_move(self, board_state: np.ndarray) -> int:
        """
        Predicts the best move for a given board state using the neural network model.

        Args:
            board_state (np.ndarray): The current game board state.

        Returns:
            int: The index of the highest probability move.
        """
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


# Example instantiation and usage
nn = NeuralNetwork(16, [64, 64], 4)
board_state = np.random.rand(1, 16)  # Example board state
predicted_moves = nn.predict(board_state)
print("Predicted move probabilities:", predicted_moves)
