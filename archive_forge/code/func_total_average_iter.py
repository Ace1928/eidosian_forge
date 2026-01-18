from __future__ import annotations
import abc
import time
from .logs import logger, null_logger
from typing import Optional, List, Dict, Any, Union
def total_average_iter(self, count: int) -> float:
    """
        Returns the average count/total duration of the timer
        """
    return count / self.total