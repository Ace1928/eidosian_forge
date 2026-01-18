import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import tyro
from typing_extensions import Annotated
from trl.trainer.utils import exact_div
from ..core import flatten_dict
from ..import_utils import is_wandb_available
TO BE FILLED In RUNTIME: the effective `batch_size` across all processes