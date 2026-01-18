import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as Fn
from xformers.components.attention import (
from xformers.components.attention.core import (

        Construct set of landmarks by recursively selecting new landmarks
        that are maximally orthogonal to the existing set.
        Returns near orthogonal landmarks with shape (B, M, D).
        