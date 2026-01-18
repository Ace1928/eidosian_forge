import os
from typing import Any
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import Checkpoint
Used to save a checkpoint on exception.

    Args:
        dirpath: directory to save the checkpoint file.
        filename: checkpoint filename. This must not include the extension.

    Raises:
        ValueError:
            If ``filename`` is empty.


    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import OnExceptionCheckpoint
        >>> trainer = Trainer(callbacks=[OnExceptionCheckpoint(".")])

    