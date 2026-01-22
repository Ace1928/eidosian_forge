import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _drop_apex_amp_state(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Apex support was removed in v2.0.0, and this migration drops it from the state-keys saved in the checkpoint
    dict.

    Version: 2.0.0
    Commit: e544676ff434ed96c6dd3b4e73a708bcb27ebcf1
    PR: #16149

    """
    key = 'amp_scaling_state'
    if key in checkpoint:
        rank_zero_warn('This checkpoint contains apex AMP data, but apex support has been removed in v2.0.0.')
        del checkpoint[key]
    return checkpoint