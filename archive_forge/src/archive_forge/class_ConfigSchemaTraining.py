import inspect
import re
from collections import defaultdict
from enum import Enum
from typing import (
from thinc.api import ConfigValidationError, Model, Optimizer
from thinc.config import Promise
from .attrs import NAMES
from .compat import Literal
from .lookups import Lookups
from .util import is_cython_func
class ConfigSchemaTraining(BaseModel):
    dev_corpus: StrictStr = Field(..., title='Path in the config to the dev data')
    train_corpus: StrictStr = Field(..., title='Path in the config to the training data')
    batcher: Batcher = Field(..., title='Batcher for the training data')
    dropout: StrictFloat = Field(..., title='Dropout rate')
    patience: StrictInt = Field(..., title='How many steps to continue without improvement in evaluation score')
    max_epochs: StrictInt = Field(..., title='Maximum number of epochs to train for')
    max_steps: StrictInt = Field(..., title='Maximum number of update steps to train for')
    eval_frequency: StrictInt = Field(..., title='How often to evaluate during training (steps)')
    seed: Optional[StrictInt] = Field(..., title='Random seed')
    gpu_allocator: Optional[StrictStr] = Field(..., title='Memory allocator when running on GPU')
    accumulate_gradient: StrictInt = Field(..., title='Whether to divide the batch up into substeps')
    score_weights: Dict[StrictStr, Optional[Union[StrictFloat, StrictInt]]] = Field(..., title='Scores to report and their weights for selecting final model')
    optimizer: Optimizer = Field(..., title='The optimizer to use')
    logger: Logger = Field(..., title='The logger to track training progress')
    frozen_components: List[str] = Field(..., title="Pipeline components that shouldn't be updated during training")
    annotating_components: List[str] = Field(..., title='Pipeline components that should set annotations during training')
    before_to_disk: Optional[Callable[['Language'], 'Language']] = Field(..., title="Optional callback to modify nlp object after training, before it's saved to disk")
    before_update: Optional[Callable[['Language', Dict[str, Any]], None]] = Field(..., title='Optional callback that is invoked at the start of each training step')

    class Config:
        extra = 'forbid'
        arbitrary_types_allowed = True