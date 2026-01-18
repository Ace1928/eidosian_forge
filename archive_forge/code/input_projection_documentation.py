import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn

    Handle all the input projections in one go, opportunistically fuse some operations.
    