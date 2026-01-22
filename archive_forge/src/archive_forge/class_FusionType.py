from enum import Enum
from functools import reduce
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from parlai.agents.transformer.modules import (
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
class FusionType(Enum):
    """
    Encoder fusion type.
    """
    EARLY = 'early'
    LATE = 'late'