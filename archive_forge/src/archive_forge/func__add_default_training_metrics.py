import copy
import logging
import sys
import warnings
from .event_handler import MetricHandler, ValidationHandler, LoggingHandler, StoppingHandler, GradientUpdateHandler
from .event_handler import TrainBegin, EpochBegin, BatchBegin, BatchEnd, EpochEnd, TrainEnd
from .event_handler import _check_event_handlers
from .utils import _check_metrics, _suggest_metric_for_loss, _check_handler_metric_ref
from ...data import DataLoader
from ...loss import Loss as gluon_loss
from ...trainer import Trainer
from ...utils import split_and_load
from ....context import Context, cpu, gpu, num_gpus
from ....metric import Loss as metric_loss
from .batch_processor import BatchProcessor
def _add_default_training_metrics(self):
    if not self._train_metrics:
        suggested_metric = _suggest_metric_for_loss(self.loss)
        if suggested_metric:
            self._train_metrics = [suggested_metric]
        loss_name = self.loss.name.rstrip('1234567890')
        self._train_metrics.append(metric_loss(loss_name))
    for metric in self._train_metrics:
        metric.name = 'training ' + metric.name