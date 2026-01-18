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
@property
def train_metrics(self):
    return self._train_metrics