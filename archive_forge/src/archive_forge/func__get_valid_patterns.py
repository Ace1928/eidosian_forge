import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools
def _get_valid_patterns(op_pattern):
    """Return a list of valid patterns generated from the op_pattern.

    Returns a list of valid patterns generated from the op_pattern,
    since MatchAllNode can match all types of nodes,
    e.g. pattern (torch.nn.Conv2d, torch.add) should also be able to match keys like
    (MatchAllNode, torch.add) and (torch.nn.Conv2d, MatchAllNode)

    Example Input:
    (torch.add, (torch.nn.ReLU, torch.nn.Conv2d))

    Example Output:
    [(torch.add, (torch.nn.ReLU, torch.nn.Conv2d)),
     (torch.add, (torch.nn.ReLU, MatchAllNode)),
     (torch.add, (MatchAllNode, torch.nn.Conv2d)),
     (torch.add, (MatchAllNode, MatchAllNode)),
     (MatchAllNode, (torch.nn.ReLU, torch.nn.Conv2d)),
     (MatchAllNode, (torch.nn.ReLU, MatchAllNode)),
     (MatchAllNode, (MatchAllNode, torch.nn.Conv2d)),
     (MatchAllNode, (MatchAllNode, MatchAllNode)),
    ]
    """
    result: List[Any]
    if isinstance(op_pattern, (tuple, list)):
        sub_combs = []
        for sub_pattern in op_pattern:
            sub_combs.append(_get_valid_patterns(sub_pattern))
        result = list(itertools.product(*sub_combs))
    else:
        result = [op_pattern, MatchAllNode]
    return result