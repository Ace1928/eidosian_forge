import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional
from ..core import flatten_dict
from ..import_utils import is_bitsandbytes_available, is_torchvision_available
Comma-separated list of prompts to use as negative examples.