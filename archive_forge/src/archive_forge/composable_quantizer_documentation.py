from __future__ import annotations
from typing import Dict, List
import torch
from torch.fx import Node
from .quantizer import QuantizationAnnotation, Quantizer
just handling global spec for now