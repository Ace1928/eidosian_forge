from typing import TYPE_CHECKING, Optional
from wandb import errors
class WaitTimeoutError(errors.Error):
    """Raised when wait() timeout occurs before process is finished."""