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
class LandmarkSelection(str, Enum):
    Orthogonal = 'orthogonal'
    KMeans = 'kmeans'
    KMeans_Spherical = 'kmeans_spherical'
    Random = 'random'