from ai_logic import (
from gui_utils import (
from game_manager import (
from typing import List, Tuple
import types
import importlib.util
import logging
import numpy as np
@StandardDecorator()
def ai_process_move(board: np.ndarray, depth: int=3) -> Tuple[np.ndarray, int]:
    """
    Processes a move on the game board using an AI agent.

    Args:
        board (np.ndarray): The current game board.
        depth (int): The depth of the search tree for the AI agent.

    Returns:
        Tuple[np.ndarray, int]: The updated game board and the total score.
    """
    return ai_game_loop(board, depth)