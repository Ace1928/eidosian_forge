import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
def _new_step(self):
    """Internal method that resets the variable for a new step."""
    self.should_save = False
    self.should_evaluate = False
    self.should_log = False