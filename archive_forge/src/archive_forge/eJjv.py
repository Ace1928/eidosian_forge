import numpy as np  # Assuming NumPy is used for efficient array manipulation
import random
from typing import List, Tuple
import types
import importlib.util
import logging


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
def expectimax(board: np.ndarray, depth: int, playerTurn: bool) -> Tuple[float, str]:
    """
    Performs the expectimax search on a given board state to a specified depth.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search.
        playerTurn (bool): Flag indicating whether it's the player's turn or a chance node.

    Returns:
        Tuple[float, str]: The best heuristic value found and the corresponding move.
    """
    if depth == 0 or is_game_over(board):
        return heuristic_evaluation(board), ""

    if playerTurn:
        best_value = float("-inf")
        best_move = ""
        for move in ["up", "down", "left", "right"]:
            new_board, _ = simulate_move(board, move)
            value, _ = expectimax(new_board, depth - 1, False)
            if value > best_value:
                best_value = value
                best_move = move
        return best_value, best_move
    else:
        total_value = 0
        empty_tiles = get_empty_tiles(board)
        num_empty = len(empty_tiles)
        if num_empty == 0:
            return heuristic_evaluation(board), ""
        for i, j in empty_tiles:
            for val in [2, 4]:
                new_board = np.array(board)
                new_board[i, j] = val
                value, _ = expectimax(new_board, depth - 1, True)
                if val == 2:
                    total_value += value * 0.9
                else:
                    total_value += value * 0.1
        return total_value / num_empty, ""


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
    smoothness, monotonicity = 0, 0
    # Implement logic for calculating smoothness and monotonicity
    # Placeholder for actual logic
    return empty_tiles * 2.7 + np.log(max_tile) * 0.9 + smoothness + monotonicity


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
        combined.extend([0] * (len(row) - len(combined)))  # Fill the rest with zeros
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
    _, best_move = expectimax(board, depth=3, playerTurn=True)
    return best_move


# Additional AI Logic, learning, optimisation and memory functions would also be defined here.
