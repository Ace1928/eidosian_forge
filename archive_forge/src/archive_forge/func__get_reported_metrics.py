import shutil
from typing import Dict, List, Optional, Union
from tensorflow.keras.callbacks import Callback as KerasCallback
import ray
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util.annotations import PublicAPI
def _get_reported_metrics(self, logs: Dict) -> Dict:
    assert isinstance(self._metrics, (type(None), str, list, dict))
    if self._metrics is None:
        reported_metrics = logs
    elif isinstance(self._metrics, str):
        reported_metrics = {self._metrics: logs[self._metrics]}
    elif isinstance(self._metrics, list):
        reported_metrics = {metric: logs[metric] for metric in self._metrics}
    elif isinstance(self._metrics, dict):
        reported_metrics = {key: logs[metric] for key, metric in self._metrics.items()}
    assert isinstance(reported_metrics, dict)
    return reported_metrics