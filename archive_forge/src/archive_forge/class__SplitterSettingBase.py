import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
class _SplitterSettingBase:

    def __init__(self, min_acc_module_size=DEFAULT_MIN_ACC_MODULE_SIZE, skip_fusion=DEFAULT_SKIP_FUSION, allow_non_tensor=DEFAULT_ALLOW_NON_TENSOR):
        parser = argparse.ArgumentParser()
        parser.add_argument('--min-acc-module-size', '--min_acc_module_size', required=False, type=int, help='Minimum size limit of an accelerator subgraph.')
        parser.add_argument('--skip-fusion', '--skip_fusion', default=False, action='store_true', help="If true then no fusion groups. Fusion group is used to enforce no non-tensor data flow between submodules. If we don't have this constrain, setting this to false is recommended as it can reduce overhead.")
        parser.add_argument('--allow-non-tensor', '--allow_non_tensor', default=False, action='store_true', help='For some backends non-tensor data flow between cpu and them are not allowed. Therefore, if a node supported by accelerator but it has non-tensor inputs or outputs to a cpu node we would want to consider it as a cpu node during splitting. However, for some backends we might not care about non-tensor data flow and we can set this option to true to disable the functionality that prevent non-tensor data flow.')
        args, unknown = parser.parse_known_args()
        self.min_acc_module_size: int = args.min_acc_module_size if args.min_acc_module_size else min_acc_module_size
        self.skip_fusion: bool = args.skip_fusion if args.skip_fusion else skip_fusion
        self.allow_non_tensor: bool = args.allow_non_tensor if args.allow_non_tensor else allow_non_tensor