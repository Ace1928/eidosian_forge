import inspect
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Type, Union
from pytorch_lightning import Callback, Trainer, LightningModule
from ray import train
from ray.util import log_once
from ray.util.annotations import PublicAPI, Deprecated
from ray.train import Checkpoint
PyTorch Lightning report and checkpoint callback

    Saves checkpoints after each validation step. Also reports metrics to Tune,
    which is needed for checkpoint registration.

    Args:
        metrics: Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to PyTorch Lightning,
            and it will reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to PyTorch Lightning.
        filename: Filename of the checkpoint within the checkpoint
            directory. Defaults to "checkpoint".
        save_checkpoints: If True (default), checkpoints will be saved and
            reported to Ray. If False, only metrics will be reported.
        on: When to trigger checkpoint creations and metric reports. Must be one of
            the PyTorch Lightning event hooks (less the ``on_``), e.g.
            "train_batch_start", or "train_end". Defaults to "validation_end".


    Example:

    .. code-block:: python

        import pytorch_lightning as pl
        from ray.tune.integration.pytorch_lightning import (
            TuneReportCheckpointCallback)

        # Save checkpoint after each training batch and after each
        # validation epoch.
        trainer = pl.Trainer(callbacks=[TuneReportCheckpointCallback(
            metrics={"loss": "val_loss", "mean_accuracy": "val_acc"},
            filename="trainer.ckpt", on="validation_end")])


    