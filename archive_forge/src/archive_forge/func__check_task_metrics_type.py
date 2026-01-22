from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from torch import Tensor, nn
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
@staticmethod
def _check_task_metrics_type(task_metrics: Dict[str, Union[Metric, MetricCollection]]) -> None:
    if not isinstance(task_metrics, dict):
        raise TypeError(f'Expected argument `task_metrics` to be a dict. Found task_metrics = {task_metrics}')
    for metric in task_metrics.values():
        if not isinstance(metric, (Metric, MetricCollection)):
            raise TypeError(f"Expected each task's metric to be a Metric or a MetricCollection. Found a metric of type {type(metric)}")