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
class MLFlowLoggerBase(MetricLogger):

    def __init__(self, client: MlflowClient):
        super().__init__()
        self._client = client

    def __getstate__(self) -> Dict[str, Any]:
        raise TuneRuntimeError(str(type(self)) + ' is not serializable')

    @property
    def client(self) -> MlflowClient:
        return self._client

    @property
    def tracking_uri(self) -> Optional[str]:
        return self.client._tracking_client.tracking_uri

    @property
    def registry_uri(self) -> Optional[str]:
        return self.client._registry_uri

    @property
    def run_id(self) -> str:
        raise TuneRuntimeError(str(type(self)) + " can't be directly used for logging")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self.client.log_metric(self.run_id, k, v)

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.client.log_param(self.run_id, k, v)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        raise TuneRuntimeError(str(type(self)) + " can't have a child logger")