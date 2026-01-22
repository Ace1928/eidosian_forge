import os
import time
import warnings
import numpy as np
from ....metric import CompositeEvalMetric, EvalMetric
from ....metric import Loss as metric_loss
from .utils import _check_metrics
class EarlyStoppingHandler(TrainBegin, EpochEnd, TrainEnd):
    """Early stop training if monitored value is not improving

    Parameters
    ----------
    monitor: EvalMetric
        The metric to monitor, and stop training if this metric does not improve.
    min_delta: float, default 0
        Minimal change in monitored value to be considered as an improvement.
    patience: int, default 0
        Number of epochs to wait for improvement before terminate training.
    mode: str, default 'auto'
        One of {auto, min, max}, if `save_best_only=True`, the comparison to make
        and determine if the monitored value has improved. if 'auto' mode, checkpoint
        handler will try to use min or max based on the monitored metric name.
    baseline: float
        Baseline value to compare the monitored value with.
    """

    def __init__(self, monitor, min_delta=0, patience=0, mode='auto', baseline=None):
        super(EarlyStoppingHandler, self).__init__()
        if not isinstance(monitor, EvalMetric):
            raise ValueError('Please provide one of the metric objects from estimator.train_metrics and estimator.val_metrics as monitor.')
        if isinstance(monitor, CompositeEvalMetric):
            raise ValueError('CompositeEvalMetric is not supported for EarlyStoppingHandler, please specify a simple metric instead.')
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.stop_training = False
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, fallback to auto mode. CheckpointHandler will usemax mode for f1 and accuracy metric comparison and use min mode other wise' % mode, RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        elif 'acc' or 'f1' in self.monitor.get()[0].lower():
            warnings.warn("`greater` operator will be used to determine if {} has improved. Please specify `mode='min'` to use the `less` operator. Specify `mode='max' to disable this warning.`".format(self.monitor.get()[0]))
            self.monitor_op = np.greater
        else:
            warnings.warn("`less` operator will be used to determine if {} has improved. Please specify `mode='max'` to use the `greater` operator. Specify `mode='min' to disable this warning.`".format(self.monitor.get()[0]))
            self.monitor_op = np.less
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def train_begin(self, estimator, *args, **kwargs):
        self.wait = 0
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.stop_training = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_end(self, estimator, *args, **kwargs):
        monitor_name, monitor_value = self.monitor.get()
        if np.isnan(monitor_value):
            warnings.warn(RuntimeWarning('%s is not updated, make sure you pass one of the metric objects fromestimator.train_metrics and estimator.val_metrics as monitor.', monitor_name))
        elif self.monitor_op(monitor_value - self.min_delta, self.best):
            self.best = monitor_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.current_epoch
                self.stop_training = True
        self.current_epoch += 1
        return self.stop_training

    def train_end(self, estimator, *args, **kwargs):
        if self.stopped_epoch > 0:
            estimator.logger.info('[Epoch %d] EarlyStoppingHanlder: early stopping due to %s not improving', self.stopped_epoch, self.monitor.get()[0])