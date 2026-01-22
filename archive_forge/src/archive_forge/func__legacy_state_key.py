from typing import Any, Dict, Type
from torch import Tensor
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
@property
def _legacy_state_key(self) -> Type['Callback']:
    """State key for checkpoints saved prior to version 1.5.0."""
    return type(self)