import os
import sys
from typing import IO, TYPE_CHECKING, Optional
from wandb.errors import CommError
class ProgressFn(Protocol):

    def __call__(self, new_bytes: int, total_bytes: int) -> None:
        pass