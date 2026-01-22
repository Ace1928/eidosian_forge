import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def _optimize_runtime_with_given_memory(memory: torch.Tensor, runtimes: torch.Tensor, max_memory: float, view_like_ops: List[int], inplace_ops: List[Tuple[int, ...]], random_ops: List[int], force_store_random: bool) -> torch.Tensor:
    """
    Given a list of operator names, their corresponding runtimes, and the maximum amount of memory available,
    find the subset of operators that can be optimized to reduce runtime while still fitting within the memory budget.
    Uses https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

    Args:
        memory (torch.Tensor): Tensor containing the memory usage of each operator.
        runtimes (torch.Tensor): Tensor containing the runtime of each operator.
        max_memory (float): Maximum amount of memory to use.
        view_like_ops ([List[int]): Indices of the view-like ops.
        inplace_ops (List[Tuple[int, int]]): Tuple with the pair of inplace op -> parent of inplace op.
            This will be used to add the constraint that in-place ops need to either be
            stored in memory with the previous op, or recomputed with the previous op.
        random_ops ([List[int]): Indices of the random ops, which will always be recomputed.
        force_store_random (bool): force random ops to always be stored (instead of recomputed)
    """
    c = -runtimes
    memory_constraint = LinearConstraint(A=memory, ub=max_memory)
    constraints = [memory_constraint]
    for i in view_like_ops:
        A = torch.zeros_like(c)
        A[i] = 1
        constraints.append(LinearConstraint(A=A, lb=0, ub=0))
    for op, op_parent in inplace_ops:
        A = torch.zeros_like(c)
        if op != op_parent:
            A[op_parent] = 1
            A[op] = -1
            constraints.append(LinearConstraint(A=A, lb=0, ub=0))
        else:
            A[op] = 1
            constraints.append(LinearConstraint(A=A, lb=1, ub=1))
    for i in random_ops:
        A = torch.zeros_like(c)
        A[i] = 1
        val = int(force_store_random)
        constraints.append(LinearConstraint(A=A, lb=val, ub=val))
    integrality = torch.ones_like(c)
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1))
    if not res.success:
        raise ValueError('The problem is infeasible, and probably due to a change in xformers that makes random ops always be stored. Try passing a larger memory_budget. This will be fixed once https://github.com/pytorch/pytorch/issues/121212 is solved')
    x = torch.from_numpy(res.x)
    return x