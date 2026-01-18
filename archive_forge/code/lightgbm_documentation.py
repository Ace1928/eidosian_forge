from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
Creates a callback that reports metrics and checkpoints model.

    Saves checkpoints after each validation step. Also reports metrics to Tune,
    which is needed for checkpoint registration.

    Args:
        metrics: Metrics to report to Tune. If this is a list,
            each item describes the metric key reported to LightGBM,
            and it will be reported under the same name to Tune. If this is a
            dict, each key will be the name reported to Tune and the respective
            value will be the metric key reported to LightGBM.
        filename: Filename of the checkpoint within the checkpoint
            directory. Defaults to "checkpoint". If this is None,
            all metrics will be reported to Tune under their default names as
            obtained from LightGBM.
        frequency: How often to save checkpoints. Defaults to 0 (no checkpoints
            are saved during training). A checkpoint is always saved at the end
            of training.
        results_postprocessing_fn: An optional Callable that takes in
            the dict that will be reported to Tune (after it has been flattened)
            and returns a modified dict that will be reported instead.

    Example:

    .. code-block:: python

        import lightgbm
        from ray.tune.integration.lightgbm import (
            TuneReportCheckpointCallback
        )

        config = {
            # ...
            "metric": ["binary_logloss", "binary_error"],
        }

        # Report only log loss to Tune after each validation epoch.
        # Save model as `lightgbm.mdl`.
        bst = lightgbm.train(
            config,
            train_set,
            valid_sets=[test_set],
            valid_names=["eval"],
            verbose_eval=False,
            callbacks=[TuneReportCheckpointCallback(
                {"loss": "eval-binary_logloss"}, "lightgbm.mdl)])

    