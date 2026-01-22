from pathlib import Path
from typing import Any, List, Tuple, Union
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Checkpoint
def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict) -> List[Tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.

    """
    checkpoints = {}
    if hasattr(checkpoint_callback, 'last_model_path') and hasattr(checkpoint_callback, 'current_score'):
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, 'latest')
    if hasattr(checkpoint_callback, 'best_model_path') and hasattr(checkpoint_callback, 'best_model_score'):
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, 'best')
    if hasattr(checkpoint_callback, 'best_k_models'):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, 'best_k')
    checkpoints = sorted(((Path(p).stat().st_mtime, p, s, tag) for p, (s, tag) in checkpoints.items() if Path(p).is_file()))
    checkpoints = [c for c in checkpoints if c[1] not in logged_model_time or logged_model_time[c[1]] < c[0]]
    return checkpoints