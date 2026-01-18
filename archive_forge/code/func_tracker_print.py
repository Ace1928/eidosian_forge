import logging
import warnings
from enum import IntEnum, unique
from typing import Any, Callable, List, Optional, TypeVar
import numpy as np
from . import collective
def tracker_print(msg: Any) -> None:
    """Print message to the tracker.
    This function can be used to communicate the information of
    the progress to the tracker
    Parameters
    ----------
    msg : str
        The message to be printed to tracker.
    """
    collective.communicator_print(msg)