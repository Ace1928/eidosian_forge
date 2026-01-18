import numpy as np
from ai_logic import (
from gui_utils import (
from typing import List, Tuple
import types
import importlib.util
import logging
import random
@StandardDecorator()
def update_game_state(board: np.ndarray, move: str) -> Tuple[np.ndarray, int]:
    """
    Updates the game state based on the move.

    Args:
        board (np.ndarray): The current game board.
        move (str): The move to be made ('up', 'down', 'left', 'right').

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the score gained from the move.
    """
    new_board, score = simulate_move(board, move)
    add_random_tile(new_board)
    return (new_board, score)