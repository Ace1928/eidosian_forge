import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class GradientUpdateHandler(BatchEnd):
    """Gradient Update Handler that apply gradients on network weights

    :py:class:`GradientUpdateHandler` takes the priority level. It updates weight parameters
    at the end of each batch

    Parameters
    ----------
    priority : scalar, default -2000
        priority level of the gradient update handler. Priority level is sorted in ascending
        order. The lower the number is, the higher priority level the handler is.
    ----------
    """

    def __init__(self, priority=-2000):
        self.priority = priority

    def batch_end(self, estimator, *args, **kwargs):
        loss = kwargs['loss']
        batch_size = 0
        if not isinstance(loss, list):
            loss = [loss]
        if isinstance(loss, list):
            for l in loss:
                batch_size += l.shape[0]
        estimator.trainer.step(batch_size)