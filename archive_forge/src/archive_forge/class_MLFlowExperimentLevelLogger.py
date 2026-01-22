import os
from typing import Any, Dict, Optional, Union
import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import (
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
class MLFlowExperimentLevelLogger(MLFlowLoggerBase):

    def __init__(self, client: MlflowClient, experiment: Experiment):
        super().__init__(client)
        self._experiment = experiment

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        assert not is_step
        return MLFlowRunLevelLogger(self, name, description=description)