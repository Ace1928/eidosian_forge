import concurrent.futures
from threading import RLock
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
Add a metric to the metric queue.

    Flush the queue if it exceeds max size.

    Args:
        key: string, the metrics key,
        value: float, the metrics value.
        step: int, the step of current metric.
        time: int, the timestamp of current metric.
        run_id: string, the run id of the associated mlflow run.
    