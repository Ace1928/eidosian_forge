import logging
import threading
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor
def _get_next_logging_step(self, run_id):
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except MlflowException:
        return 0
    system_metric_name = None
    for metric_name in run.data.metrics.keys():
        if metric_name.startswith(self._metrics_prefix):
            system_metric_name = metric_name
            break
    if system_metric_name is None:
        return 0
    metric_history = client.get_metric_history(run_id, system_metric_name)
    return metric_history[-1].step + 1