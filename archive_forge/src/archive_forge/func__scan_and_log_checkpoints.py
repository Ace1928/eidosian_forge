import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.loggers.utilities import _scan_checkpoints
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
    import wandb
    checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
    for t, p, s, tag in checkpoints:
        metadata = {'score': s.item() if isinstance(s, Tensor) else s, 'original_filename': Path(p).name, checkpoint_callback.__class__.__name__: {k: getattr(checkpoint_callback, k) for k in ['monitor', 'mode', 'save_last', 'save_top_k', 'save_weights_only', '_every_n_train_steps'] if hasattr(checkpoint_callback, k)}}
        if not self._checkpoint_name:
            self._checkpoint_name = f'model-{self.experiment.id}'
        artifact = wandb.Artifact(name=self._checkpoint_name, type='model', metadata=metadata)
        artifact.add_file(p, name='model.ckpt')
        aliases = ['latest', 'best'] if p == checkpoint_callback.best_model_path else ['latest']
        self.experiment.log_artifact(artifact, aliases=aliases)
        self._logged_model_time[p] = t