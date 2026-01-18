import logging
import os
from typing import Any, Callable, Dict, Optional
from typing_extensions import override
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.utilities.cloud_io import _atomic_save, get_filesystem
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.types import _PATH
Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        