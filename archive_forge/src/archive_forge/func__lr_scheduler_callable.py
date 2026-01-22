import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def _lr_scheduler_callable(optimizer):
    trainer_copy = copy.copy(trainer)
    trainer_copy.lr_scheduler = None
    lr_scheduler = trainer_copy.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
    return lr_scheduler