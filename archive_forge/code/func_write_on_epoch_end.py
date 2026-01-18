from typing import Any, Literal, Optional, Sequence
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def write_on_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
    """Override with the logic to write all batches."""
    raise NotImplementedError()