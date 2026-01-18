import logging
from types import ModuleType
from typing import Dict, Optional, Union
import ray
from ray.air import session
from ray.air._internal.mlflow import _MLflowLoggerUtil
from ray.air._internal import usage as air_usage
from ray.air.constants import TRAINING_ITERATION
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL
from ray.tune.experiment import Trial
from ray.util.annotations import PublicAPI
MLflow Logger to automatically log Tune results and config to MLflow.

    MLflow (https://mlflow.org) Tracking is an open source library for
    recording and querying experiments. This Ray Tune ``LoggerCallback``
    sends information (config parameters, training results & metrics,
    and artifacts) to MLflow for automatic experiment tracking.

    Keep in mind that the callback will open an MLflow session on the driver and
    not on the trainable. Therefore, it is not possible to call MLflow functions
    like ``mlflow.log_figure()`` inside the trainable as there is no MLflow session
    on the trainable. For more fine grained control, use :func:`setup_mlflow`.

    Args:
        tracking_uri: The tracking URI for where to manage experiments
            and runs. This can either be a local file path or a remote server.
            This arg gets passed directly to mlflow
            initialization. When using Tune in a multi-node setting, make sure
            to set this to a remote server and not a local file path.
        registry_uri: The registry URI that gets passed directly to
            mlflow initialization.
        experiment_name: The experiment name to use for this Tune run.
            If the experiment with the name already exists with MLflow,
            it will be reused. If not, a new experiment will be created with
            that name.
        tags: An optional dictionary of string keys and values to set
            as tags on the run
        tracking_token: Tracking token used to authenticate with MLflow.
        save_artifact: If set to True, automatically save the entire
            contents of the Tune local_dir as an artifact to the
            corresponding run in MlFlow.

    Example:

    .. code-block:: python

        from ray.air.integrations.mlflow import MLflowLoggerCallback

        tags = { "user_name" : "John",
                 "git_commit_hash" : "abc123"}

        tune.run(
            train_fn,
            config={
                # define search space here
                "parameter_1": tune.choice([1, 2, 3]),
                "parameter_2": tune.choice([4, 5, 6]),
            },
            callbacks=[MLflowLoggerCallback(
                experiment_name="experiment1",
                tags=tags,
                save_artifact=True)])

    