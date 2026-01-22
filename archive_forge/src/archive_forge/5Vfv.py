import collections
import numpy as np
from typing import Deque, Dict, Tuple
import logging
import hashlib
import pickle
import zlib

# Setting up logging for detailed insights into the memory operations
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Implementing a class to manage short-term memory for the AI. This memory stores recent moves and their outcomes.
class ShortTermMemory:
    """
    Manages the short-term memory for the AI, storing recent moves and their outcomes.

    Attributes:
        memory (Deque[Tuple[np.ndarray, str, int]]): A deque storing the recent game states, moves, and scores.
        capacity (int): The maximum capacity of the short-term memory.

    Methods:
        __init__(self, capacity: int = 10) -> None:
            Initializes a new instance of the ShortTermMemory class.
        store(self, board: np.ndarray, move: str, score: int) -> None:
            Stores the given game state, move, and score in the short-term memory.
    """

    def __init__(self, capacity: int = 10) -> None:
        """
        Initializes a new instance of the ShortTermMemory class.

        Args:
            capacity (int): The maximum capacity of the short-term memory. Defaults to 10.

        Returns:
            None

        Raises:
            None

        Example:
            >>> short_term_memory = ShortTermMemory(capacity=5)
        """
        self.memory: Deque[Tuple[np.ndarray, str, int]] = collections.deque(
            maxlen=capacity
        )
        self.capacity: int = capacity
        logging.info(f"Initialized ShortTermMemory with capacity {capacity}")

    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the short-term memory.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.

        Returns:
            None

        Raises:
            None

        Example:
            >>> board = np.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            >>> move = "right"
            >>> score = 2
            >>> short_term_memory.store(board, move, score)
        """
        self.memory.append((board, move, score))
        logging.debug(
            f"Stored in ShortTermMemory: Board: {board}, Move: {move}, Score: {score}"
        )

        # Additional logging for detailed insights
        logging.debug(f"Current ShortTermMemory size: {len(self.memory)}")
        logging.debug(f"Current ShortTermMemory contents: {self.memory}")

        # TODO: Implement additional functionality for short-term memory management
        # - Prioritize storage based on score or other criteria
        # - Implement retrieval methods for accessing stored data
        # - Integrate with other components of the AI system


# Implementing a class to manage LRU memory for the AI. Acts as a ranked working memory for game states, moves, and scores.
class LRUMemory:
    """
    Implements a Least Recently Used (LRU) memory cache to store game states, moves, and scores.

    The LRUMemory class provides a caching mechanism that keeps the most recently used items in memory while evicting the least recently used items when the capacity is exceeded.
    It uses an OrderedDict to maintain the order of items based on their usage, allowing efficient access and eviction.

    Attributes:
        capacity (int): The maximum number of items the LRUMemory can store.
        cache (Dict[Tuple[str, int], np.ndarray]): The ordered dictionary that stores the cached items, where the key is a tuple of (move, score) and the value is the game board state.

    Methods:
        __init__(self, capacity: int = 50) -> None:
            Initializes a new instance of the LRUMemory class with the specified capacity.

        store(self, board: np.ndarray, move: str, score: int) -> None:
            Stores the given game state, move, and score in the LRU memory, evicting the least recently used item if necessary.

    Example Usage:
        >>> lru_memory = LRUMemory(capacity=100)
        >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
        >>> move = "left"
        >>> score = 24
        >>> lru_memory.store(board, move, score)
    """

    def __init__(self, capacity: int = 50) -> None:
        """
        Initializes a new instance of the LRUMemory class with the specified capacity.

        Args:
            capacity (int): The maximum number of items the LRUMemory can store. Defaults to 50.

        Returns:
            None

        Raises:
            None

        Example:
            >>> lru_memory = LRUMemory(capacity=100)
        """
        self.capacity: int = capacity
        self.cache: Dict[Tuple[str, int], np.ndarray] = collections.OrderedDict()
        logging.info(f"Initialized LRUMemory with capacity {capacity}")

    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the LRU memory, evicting the least recently used item if necessary.

        Args:
            board (np.ndarray): The current game board state.
            move (str): The move made to reach the current game state.
            score (int): The score gained from the move.

        Returns:
            None

        Raises:
            None

        Example:
            >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
            >>> move = "left"
            >>> score = 24
            >>> lru_memory.store(board, move, score)
        """
        # Create a key tuple consisting of the move and score
        key: Tuple[str, int] = (move, score)

        # Check if the key already exists in the cache
        if key in self.cache:
            # If the key exists, move it to the end (most recently used)
            self.cache.move_to_end(key)
            logging.debug(f"Moved existing key {key} to the end of LRUMemory cache")

        # Store the board state in the cache with the key
        self.cache[key] = board
        logging.debug(f"Stored board state in LRUMemory cache with key {key}")

        # Check if the cache size exceeds the capacity
        if len(self.cache) > self.capacity:
            # If the cache size exceeds the capacity, evict the least recently used item
            evicted_item: Tuple[Tuple[str, int], np.ndarray] = self.cache.popitem(
                last=False
            )
            logging.debug(
                f"Evicted least recently used item from LRUMemory cache: {evicted_item}"
            )

        logging.debug(
            f"Stored in LRUMemory: Board: {board}, Move: {move}, Score: {score}"
        )

        # Additional logging for detailed insights
        logging.debug(f"Current LRUMemory cache size: {len(self.cache)}")
        logging.debug(f"Current LRUMemory cache contents: {self.cache}")

        # TODO: Implement additional functionality for LRU memory management
        # - Implement retrieval methods for accessing stored data
        # - Integrate with other components of the AI system
        # - Optimize memory usage and performance
        # - Handle edge cases and error scenarios


# Implementing a class to manage the learning strategy for the AI. Acts as a long-term memory for game states, moves, and scores.
class LongTermMemory:
    """
    Manages the long-term memory for the AI, storing relevant game states, moves, and scores for learning and optimization.
    Utilizes reversible encoding/decoding, compression and vectorization of all stored long term values.
    Using pickle for serialization and deserialization of the string representation of the vectorized data after all encoding and compression.
    Deserialized and then decompressed and then decoded back to original string from the deserialized decompressed devectorized value.
    Implementing efficient indexing and retrieval of stored data for training and decision-making.
    Utilizing a ranking system to determine the most relevant data for decision-making.
    Utilizing a mechanism to normalize and standardize data stored to long term memory to ensure no duplication or redundancy.
    Using a hashing mechanism to ensure data integrity and consistency and uniqueness of stored data.
    """

    def __init__(self, capacity: int = 100) -> None:
        """
        Initializes a new instance of the LongTermMemory class.

        Args:
            capacity (int): The maximum capacity of the long-term memory. Defaults to 100.

        Returns:
            None

        Raises:
            None

        Example:
            >>> long_term_memory = LongTermMemory(capacity=200)
        """
        self.capacity: int = capacity
        self.memory: Dict[str, Tuple[np.ndarray, str, int]] = {}
        logging.info(f"Initialized LongTermMemory with capacity {capacity}")

    def store(self, board: np.ndarray, move: str, score: int) -> None:
        """
        Stores the given game state, move, and score in the long-term memory.

        Args:
            board (np.ndarray): The current game board.
            move (str): The move made.
            score (int): The score gained from the move.

        Returns:
            None

        Raises:
            None

        Example:
            >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
            >>> move = "left"
            >>> score = 24
            >>> long_term_memory.store(board, move, score)
        """
        # Generate a hash key for the board
        key: str = self._hash(board)

        # Check if the key does not exist in the memory
        if key not in self.memory:
            # Store the board, move, and score in the memory with the key
            self.memory[key] = (board, move, score)
            logging.debug(
                f"Stored new entry in LongTermMemory: Board: {board}, Move: {move}, Score: {score}"
            )

            # Check if the memory size exceeds the capacity
            if len(self.memory) > self.capacity:
                # Remove the least relevant item from the memory
                self._remove_least_relevant()
                logging.debug(
                    f"Removed least relevant item from LongTermMemory due to capacity overflow"
                )
        else:
            logging.debug(
                f"Skipped storing duplicate entry in LongTermMemory: Board: {board}, Move: {move}, Score: {score}"
            )

        # Additional logging for detailed insights
        logging.debug(f"Current LongTermMemory size: {len(self.memory)}")
        logging.debug(f"Current LongTermMemory contents: {self.memory}")

    def _remove_least_relevant(self) -> None:
        """
        Removes the least relevant item from the long-term memory based on a ranking system.

        Returns:
            None

        Raises:
            None

        Example:
            >>> long_term_memory._remove_least_relevant()
        """
        # Rank all items in the memory based on relevance
        self._rank_all()

        # Get the least relevant item (lowest rank) from the memory
        least_relevant_key: str = min(self.memory, key=lambda x: self.memory[x][3])

        # Remove the least relevant item from the memory
        removed_item: Tuple[np.ndarray, str, int] = self.memory.pop(least_relevant_key)

        logging.debug(
            f"Removed least relevant item from LongTermMemory: {removed_item}"
        )

    def retrieve(self, board: np.ndarray) -> Tuple[np.ndarray, str, int]:
        """
        Retrieves the stored move and score for the given game board from the long-term memory.

        Args:
            board (np.ndarray): The game board.

        Returns:
            Tuple[np.ndarray, str, int]: The stored board, move, and score. Returns (None, None, None) if not found.

        Raises:
            None

        Example:
            >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
            >>> retrieved_board, retrieved_move, retrieved_score = long_term_memory.retrieve(board)
        """
        # Generate a hash key for the board
        key: str = self._hash(board)

        # Retrieve the stored board, move, and score from the memory using the key
        stored_data: Tuple[np.ndarray, str, int] = self.memory.get(
            key, (None, None, None)
        )

        logging.debug(
            f"Retrieved from LongTermMemory: Board: {stored_data[0]}, Move: {stored_data[1]}, Score: {stored_data[2]}"
        )

        return stored_data

    def _encode(self, board: np.ndarray) -> str:
        """
        Encodes the game board into a string representation for storage.

        Args:
            board (np.ndarray): The game board.

        Returns:
            str: The encoded string representation of the board.

        Raises:
            None

        Example:
            >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
            >>> encoded_board = long_term_memory._encode(board)
        """
        # Convert the board to a string representation
        encoded_board: str = np.array2string(board, separator=",")

        logging.debug(f"Encoded board: {encoded_board}")

        return encoded_board

    def _decode(self, encoded_board: str) -> np.ndarray:
        """
        Decodes the encoded string representation of the board back into a NumPy array.

        Args:
            encoded_board (str): The encoded string representation of the board.

        Returns:
            np.ndarray: The decoded game board.

        Raises:
            None

        Example:
            >>> encoded_board = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> decoded_board = long_term_memory._decode(encoded_board)
        """
        # Convert the encoded string back to a NumPy array
        decoded_board: np.ndarray = np.fromstring(
            encoded_board.replace("[", "").replace("]", ""), sep=","
        ).reshape((4, 4))

        logging.debug(f"Decoded board: {decoded_board}")

        return decoded_board

    def _compress(self, data: str) -> str:
        """
        Compresses the given data to reduce memory usage.

        Args:
            data (str): The data to compress.

        Returns:
            str: The compressed data.

        Raises:
            None

        Example:
            >>> data = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> compressed_data = long_term_memory._compress(data)
        """
        # Compress the data using zlib compression
        compressed_data: str = zlib.compress(data.encode("utf-8")).hex()

        logging.debug(f"Compressed data: {compressed_data}")

        return compressed_data

    def _decompress(self, compressed_data: str) -> str:
        """
        Decompresses the compressed data back to its original form.

        Args:
            compressed_data (str): The compressed data.

        Returns:
            str: The decompressed data.

        Raises:
            None

        Example:
            >>> compressed_data = "789c2b492d2e5170740400a0f8bf8b0d0a0d0a"
            >>> decompressed_data = long_term_memory._decompress(compressed_data)
        """
        # Decompress the data using zlib decompression
        decompressed_data: str = zlib.decompress(bytes.fromhex(compressed_data)).decode(
            "utf-8"
        )

        logging.debug(f"Decompressed data: {decompressed_data}")

        return decompressed_data

    def _vectorize(self, data: str) -> np.ndarray:
        """
        Vectorizes the data for efficient storage and retrieval.

        Args:
            data (str): The data to vectorize.

        Returns:
            np.ndarray: The vectorized data.

        Raises:
            None

        Example:
            >>> data = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> vectorized_data = long_term_memory._vectorize(data)
        """
        # Vectorize the data by converting each character to its ASCII code
        vectorized_data: np.ndarray = np.array([ord(char) for char in data])

        logging.debug(f"Vectorized data: {vectorized_data}")

        return vectorized_data

    def _devectorize(self, vectorized_data: np.ndarray) -> str:
        """
        Devectorizes the vectorized data back to its original form.

        Args:
            vectorized_data (np.ndarray): The vectorized data.

        Returns:
            str: The devectorized data.

        Raises:
            None

        Example:
            >>> vectorized_data = np.array([91, 91, 50, 32, 48, 32, 48, 32, 48, 93, 44, 32, 91, 48, 32, 52, 32, 48, 32, 48, 93, 44, 32, 91, 48, 32, 48, 32, 56, 32, 48, 93, 44, 32, 91, 48, 32, 48, 32, 48, 32, 49, 54, 93, 93])
            >>> devectorized_data = long_term_memory._devectorize(vectorized_data)
        """
        # Devectorize the data by converting each ASCII code back to its corresponding character
        devectorized_data: str = "".join([chr(int(val)) for val in vectorized_data])

        logging.debug(f"Devectorized data: {devectorized_data}")

        return devectorized_data

    def _serialize(self, data: str) -> str:
        """
        Serializes the data for storage.

        Args:
            data (str): The data to serialize.

        Returns:
            str: The serialized data.

        Raises:
            None

        Example:
            >>> data = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> serialized_data = long_term_memory._serialize(data)
        """
        # Serialize the data using pickle
        serialized_data: str = pickle.dumps(data).hex()

        logging.debug(f"Serialized data: {serialized_data}")

        return serialized_data

    def _deserialize(self, serialized_data: str) -> str:
        """
        Deserializes the serialized data back to its original form.

        Args:
            serialized_data (str): The serialized data.

        Returns:
            str: The deserialized data.

        Raises:
            None

        Example:
            >>> serialized_data = "80049528285b5b3220302030205d2c205b3020342030205d2c205b3020302038205d2c205b302030203020313629"
            >>> deserialized_data = long_term_memory._deserialize(serialized_data)
        """
        # Deserialize the data using pickle
        deserialized_data: str = pickle.loads(bytes.fromhex(serialized_data))

        logging.debug(f"Deserialized data: {deserialized_data}")

        return deserialized_data

    def _normalize(self, data: str) -> str:
        """
        Normalizes the data to ensure consistency and uniqueness.

        Args:
            data (str): The data to normalize.

        Returns:
            str: The normalized data.

        Raises:
            None

        Example:
            >>> data = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> normalized_data = long_term_memory._normalize(data)
        """
        # Normalize the data by removing whitespace and converting to lowercase
        normalized_data: str = data.replace(" ", "").lower()

        logging.debug(f"Normalized data: {normalized_data}")

        return normalized_data

    def _standardize(self, data: str) -> str:
        """
        Standardizes the data to ensure no duplication or redundancy.

        Args:
            data (str): The data to standardize.

        Returns:
            str: The standardized data.

        Raises:
            None

        Example:
            >>> data = "[[2 0 0 0], [0 4 0 0], [0 0 8 0], [0 0 0 16]]"
            >>> standardized_data = long_term_memory._standardize(data)
        """
        # Standardize the data by sorting the elements within each row and column
        standardized_data: str = str(np.sort(np.sort(eval(data), axis=0), axis=1))

        logging.debug(f"Standardized data: {standardized_data}")

        return standardized_data

    def _hash(self, board: np.ndarray) -> str:
        """
        Generates a hash key for the given game board.

        Args:
            board (np.ndarray): The game board.

        Returns:
            str: The hashed key for the board.

        Raises:
            None

        Example:
            >>> board = np.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]])
            >>> hashed_key = long_term_memory._hash(board)
        """
        # Generate a hash key by converting the board to a string and hashing it
        hashed_key: str = hashlib.sha256(str(board).encode("utf-8")).hexdigest()

        logging.debug(f"Hashed key: {hashed_key}")

        return hashed_key

    def _unhash(self, key: str) -> np.ndarray:
        """
        Retrieves the game board from the hashed key.

        Args:
            key (str): The hashed key for the board.

        Returns:
            np.ndarray: The game board.

        Raises:
            None

        Example:
            >>> hashed_key = "1f4e4c4f9b4d4b4c9b4d4b4c9b4d4b4c9b4d4b4c"
            >>> board = long_term_memory._unhash(hashed_key)
        """
        # Retrieve the board from the memory using the hashed key
        return np.frombuffer(key)

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

    def _rank(self, data: str) -> int:
        """
        Ranks the data based on relevance for decision-making.

        Args:
            data (str): The data to rank.

        Returns:
            int: The ranking value.
        """
        # Get the rank of a single item utilising the rank all item to sort the data in the store
        return self.memory[data][2]

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
