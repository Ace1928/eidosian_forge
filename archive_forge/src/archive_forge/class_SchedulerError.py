import logging
from typing import Any, Callable, Dict
class SchedulerError(Exception):
    """Raised when a known error occurs with wandb sweep scheduler."""
    pass