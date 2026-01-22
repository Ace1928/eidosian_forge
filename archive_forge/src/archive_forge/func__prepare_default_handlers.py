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
def _prepare_default_handlers(self, val_data, event_handlers):
    event_handlers = _check_event_handlers(event_handlers)
    added_default_handlers = []
    added_default_handlers.append(StoppingHandler(self.max_epoch, self.max_batch))
    if not any((isinstance(handler, GradientUpdateHandler) for handler in event_handlers)):
        added_default_handlers.append(GradientUpdateHandler())
    if not any((isinstance(handler, MetricHandler) for handler in event_handlers)):
        added_default_handlers.append(MetricHandler(metrics=self.train_metrics))
    if not any((isinstance(handler, ValidationHandler) for handler in event_handlers)):
        if val_data:
            added_default_handlers.append(ValidationHandler(val_data=val_data, eval_fn=self.evaluate))
    if not any((isinstance(handler, LoggingHandler) for handler in event_handlers)):
        added_default_handlers.append(LoggingHandler(metrics=self.train_metrics))
    mixing_handlers = event_handlers and added_default_handlers
    event_handlers.extend(added_default_handlers)
    if mixing_handlers:
        known_metrics = set(self.train_metrics + self.val_metrics)
        for handler in event_handlers:
            _check_handler_metric_ref(handler, known_metrics)
    event_handlers.sort(key=lambda handler: getattr(handler, 'priority', 0))
    return event_handlers