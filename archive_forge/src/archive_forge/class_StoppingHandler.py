import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class StoppingHandler(TrainBegin, BatchEnd, EpochEnd):
    """Stop conditions to stop training
    Stop training if maximum number of batches or epochs
    reached.

    Parameters
    ----------
    max_epoch : int, default None
        Number of maximum epochs to train.
    max_batch : int, default None
        Number of maximum batches to train.

    """

    def __init__(self, max_epoch=None, max_batch=None):
        self.max_epoch = max_epoch
        self.max_batch = max_batch
        self.current_batch = 0
        self.current_epoch = 0
        self.stop_training = False

    def train_begin(self, estimator, *args, **kwargs):
        self.max_epoch = estimator.max_epoch
        self.max_batch = estimator.max_batch
        self.current_batch = 0
        self.current_epoch = 0

    def batch_end(self, estimator, *args, **kwargs):
        self.current_batch += 1
        if self.current_batch == self.max_batch:
            self.stop_training = True
        return self.stop_training

    def epoch_end(self, estimator, *args, **kwargs):
        self.current_epoch += 1
        if self.current_epoch == self.max_epoch:
            self.stop_training = True
        return self.stop_training