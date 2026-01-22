import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional, Union
from packaging import version
from typing_extensions import override
import wandb
from wandb import Artifact
from wandb.sdk.lib import RunDisabled, telemetry
from wandb.sdk.wandb_run import Run
@override
def after_save_checkpoint(self, checkpoint_callback: 'ModelCheckpoint') -> None:
    if self._log_model == 'all' or (self._log_model is True and checkpoint_callback.save_top_k == -1):
        self._scan_and_log_pytorch_checkpoints(checkpoint_callback)
    elif self._log_model is True:
        self._checkpoint_callback = checkpoint_callback