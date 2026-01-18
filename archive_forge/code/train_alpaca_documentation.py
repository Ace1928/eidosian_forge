import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
Make dataset and collator for supervised fine-tuning.