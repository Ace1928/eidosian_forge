import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
class DagsHubCallback(MLflowCallback):
    """
    A [`TrainerCallback`] that logs to [DagsHub](https://dagshub.com/). Extends [`MLflowCallback`]
    """

    def __init__(self):
        super().__init__()
        if not is_dagshub_available():
            raise ImportError('DagsHubCallback requires dagshub to be installed. Run `pip install dagshub`.')
        from dagshub.upload import Repo
        self.Repo = Repo

    def setup(self, *args, **kwargs):
        """
        Setup the DagsHub's Logging integration.

        Environment:
        - **HF_DAGSHUB_LOG_ARTIFACTS** (`str`, *optional*):
                Whether to save the data and model artifacts for the experiment. Default to `False`.
        """
        self.log_artifacts = os.getenv('HF_DAGSHUB_LOG_ARTIFACTS', 'FALSE').upper() in ENV_VARS_TRUE_VALUES
        self.name = os.getenv('HF_DAGSHUB_MODEL_NAME') or 'main'
        self.remote = os.getenv('MLFLOW_TRACKING_URI')
        self.repo = self.Repo(owner=self.remote.split(os.sep)[-2], name=self.remote.split(os.sep)[-1].split('.')[0], branch=os.getenv('BRANCH') or 'main')
        self.path = Path('artifacts')
        if self.remote is None:
            raise RuntimeError('DagsHubCallback requires the `MLFLOW_TRACKING_URI` environment variable to be set. Did you run `dagshub.init()`?')
        super().setup(*args, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_artifacts:
            if getattr(self, 'train_dataloader', None):
                torch.save(self.train_dataloader.dataset, os.path.join(args.output_dir, 'dataset.pt'))
            self.repo.directory(str(self.path)).add_dir(args.output_dir)