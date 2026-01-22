import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
class TorchSplit(CallFunction):
    """
    Matches a call to torch.split if it is in a normalized form. Ensures that all users of
    splits are unique getitems.
    """

    def __init__(self, arg, sizes):
        super().__init__(torch.split, arg, sizes, _users=MULTIPLE, dim=KeywordArg('dim'))

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = super()._match(node, ctx)
        if not m:
            return m
        split_sections = node.args[1]
        if not isinstance(split_sections, (list, tuple)):
            return FailedMatch('split not normalized')
        seen_idxs = set()
        for user in node.users:
            if not CallFunction(operator.getitem, Arg(), Arg()).match(user):
                return FailedMatch(f'user of split not a getitem: {user}')
            if not isinstance(user.args[1], int):
                return FailedMatch('only integer getitems are handled')
            if user.args[1] in seen_idxs:
                return FailedMatch(f'duplicate getitem {user.args[1]}')
            if user.args[-1] < 0:
                return FailedMatch('negative index')
            seen_idxs.add(user.args[1])
        return m