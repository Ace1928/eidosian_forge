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
class MLflowLoggerCallback(LoggerCallback):
    """MLflow Logger to automatically log Tune results and config to MLflow.

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

    """

    def __init__(self, tracking_uri: Optional[str]=None, *, registry_uri: Optional[str]=None, experiment_name: Optional[str]=None, tags: Optional[Dict]=None, tracking_token: Optional[str]=None, save_artifact: bool=False):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        self.tags = tags
        self.tracking_token = tracking_token
        self.should_save_artifact = save_artifact
        self.mlflow_util = _MLflowLoggerUtil()
        if ray.util.client.ray.is_connected():
            logger.warning('When using MLflowLoggerCallback with Ray Client, it is recommended to use a remote tracking server. If you are using a MLflow tracking server backed by the local filesystem, then it must be setup on the server side and not on the client side.')

    def setup(self, *args, **kwargs):
        self.mlflow_util.setup_mlflow(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri, experiment_name=self.experiment_name, tracking_token=self.tracking_token)
        if self.tags is None:
            self.tags = {}
        self._trial_runs = {}

    def log_trial_start(self, trial: 'Trial'):
        if trial not in self._trial_runs:
            tags = self.tags.copy()
            tags['trial_name'] = str(trial)
            run = self.mlflow_util.start_run(tags=tags, run_name=str(trial))
            self._trial_runs[trial] = run.info.run_id
        run_id = self._trial_runs[trial]
        config = trial.config
        self.mlflow_util.log_params(run_id=run_id, params_to_log=config)

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        run_id = self._trial_runs[trial]
        self.mlflow_util.log_metrics(run_id=run_id, metrics_to_log=result, step=step)

    def log_trial_end(self, trial: 'Trial', failed: bool=False):
        run_id = self._trial_runs[trial]
        if self.should_save_artifact:
            self.mlflow_util.save_artifacts(run_id=run_id, dir=trial.local_path)
        status = 'FINISHED' if not failed else 'FAILED'
        self.mlflow_util.end_run(run_id=run_id, status=status)