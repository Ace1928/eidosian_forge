import sys
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type
import wandb
from wandb.errors import Error
def was_ctrl_c(self) -> bool:
    return isinstance(self.exception, KeyboardInterrupt)