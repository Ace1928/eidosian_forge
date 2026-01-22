import copy
import functools
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import (
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import (
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import (
def _is_quantized_op_pt2e(node: torch.fx.Node):
    """
    Used for pt2e flow to check if the node is a quantized node:
    Case1: the node has been annotated as output node of a fusion pattern.
    Case2: the node has been annotated as single quantized node.
    """
    if not _is_any_annotated([node]):
        return False
    quantization_annotation = node.meta.get(QUANT_ANNOTATION_KEY, None)
    assert isinstance(quantization_annotation, _X86InductorQuantizationAnnotation)
    return quantization_annotation._is_output_of_quantized_pattern