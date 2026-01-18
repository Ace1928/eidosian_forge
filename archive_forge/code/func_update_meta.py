import random
import shutil
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import (
from thinc.api import Config, Optimizer, constant, fix_random_seed, set_gpu_allocator
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaTraining
from ..util import logger, registry, resolve_dot_names
from .example import Example
def update_meta(training: Union[Dict[str, Any], Config], nlp: 'Language', info: Dict[str, Any]) -> None:
    nlp.meta['performance'] = {}
    for metric in training['score_weights']:
        if metric is not None:
            nlp.meta['performance'][metric] = info['other_scores'].get(metric, 0.0)
    for pipe_name in nlp.pipe_names:
        if pipe_name in info['losses']:
            nlp.meta['performance'][f'{pipe_name}_loss'] = info['losses'][pipe_name]