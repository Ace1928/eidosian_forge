from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
from huggingface_hub._commit_scheduler import CommitScheduler
from .utils import experimental, is_tensorboard_available
Push to hub in a non-blocking way when exiting the logger's context manager.