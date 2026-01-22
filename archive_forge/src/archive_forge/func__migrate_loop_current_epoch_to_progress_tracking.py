import re
from typing import Any, Callable, Dict, List
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _migrate_loop_current_epoch_to_progress_tracking(checkpoint: _CHECKPOINT) -> _CHECKPOINT:
    """Sets the `current_epoch` value for checkpoints before v1.6 without the progress tracking state. It will be
    overwritten by the loop's state if it was also saved.

    Version: 1.6.0
    Commit: aea96e4
    PR: #11805

    """
    epoch = checkpoint['epoch']
    checkpoint.setdefault('loops', {'fit_loop': _get_fit_loop_initial_state_1_6_0()})
    checkpoint['loops'].setdefault('fit_loop', _get_fit_loop_initial_state_1_6_0())
    checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed'] = epoch
    return checkpoint