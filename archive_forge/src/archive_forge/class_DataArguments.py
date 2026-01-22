import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={'help': 'Path to the training data.'})