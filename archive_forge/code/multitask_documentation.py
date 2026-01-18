from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from torch import Tensor, nn
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
Plot a single or multiple values from the metric.

        All tasks' results are plotted on individual axes.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            axes: Sequence of matplotlib axis objects. If provided, will add the plots to the provided axis objects.
                If not provided, will create them.

        Returns:
            Sequence of tuples with Figure and Axes object for each task.

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.wrappers import MultitaskWrapper
            >>> from torchmetrics.regression import MeanSquaredError
            >>> from torchmetrics.classification import BinaryAccuracy
            >>>
            >>> classification_target = torch.tensor([0, 1, 0])
            >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = torch.tensor([0, 0, 1])
            >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({
            ...     "Classification": BinaryAccuracy(),
            ...     "Regression": MeanSquaredError()
            ... })
            >>> metrics.update(preds, targets)
            >>> value = metrics.compute()
            >>> fig_, ax_ = metrics.plot(value)

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.wrappers import MultitaskWrapper
            >>> from torchmetrics.regression import MeanSquaredError
            >>> from torchmetrics.classification import BinaryAccuracy
            >>>
            >>> classification_target = torch.tensor([0, 1, 0])
            >>> regression_target = torch.tensor([2.5, 5.0, 4.0])
            >>> targets = {"Classification": classification_target, "Regression": regression_target}
            >>>
            >>> classification_preds = torch.tensor([0, 0, 1])
            >>> regression_preds = torch.tensor([3.0, 5.0, 2.5])
            >>> preds = {"Classification": classification_preds, "Regression": regression_preds}
            >>>
            >>> metrics = MultitaskWrapper({
            ...     "Classification": BinaryAccuracy(),
            ...     "Regression": MeanSquaredError()
            ... })
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metrics(preds, targets))
            >>> fig_, ax_ = metrics.plot(values)

        