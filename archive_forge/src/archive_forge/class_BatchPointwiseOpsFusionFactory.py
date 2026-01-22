import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
class BatchPointwiseOpsFusionFactory(BatchFusion):

    def __init__(self, op, **kwargs):
        super().__init__(**kwargs)
        self.op = op