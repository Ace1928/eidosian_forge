import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class TrainEnd(EventHandler):

    def train_end(self, estimator, *args, **kwargs):
        pass