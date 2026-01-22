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
def _categorize_handlers(self, event_handlers):
    """
        categorize handlers into 6 event lists to avoid calling empty methods
        for example, only event handlers with train_begin method
        implemented will be called at train begin
        """
    train_begin = []
    epoch_begin = []
    batch_begin = []
    batch_end = []
    epoch_end = []
    train_end = []
    for handler in event_handlers:
        if isinstance(handler, TrainBegin):
            train_begin.append(handler)
        if isinstance(handler, EpochBegin):
            epoch_begin.append(handler)
        if isinstance(handler, BatchBegin):
            batch_begin.append(handler)
        if isinstance(handler, BatchEnd):
            batch_end.append(handler)
        if isinstance(handler, EpochEnd):
            epoch_end.append(handler)
        if isinstance(handler, TrainEnd):
            train_end.append(handler)
    return (train_begin, epoch_begin, batch_begin, batch_end, epoch_end, train_end)