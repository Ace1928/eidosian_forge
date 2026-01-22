import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
@torch.jit.ignore
def _get_minimal_slice_set(start: Sequence[int], end: Sequence[int], dims: Sequence[int], start_edges: Optional[Sequence[bool]]=None, end_edges: Optional[Sequence[bool]]=None) -> List[Tuple[slice, ...]]:
    """
    Produces an ordered sequence of tensor slices that, when used in sequence on a tensor with shape dims, yields
    tensors that contain every leaf in the contiguous range [start, end]. Care is taken to yield a short sequence of
    slices, and perhaps even the shortest possible (I'm pretty sure it's the latter).

    end is INCLUSIVE.
    """

    def reduce_edge_list(l: List[bool]) -> None:
        tally = True
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] &= tally
            tally = l[reversed_idx]
    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == d - 1 for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)
    if len(start) == 0:
        return [()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]
    slices: List[Tuple[slice, ...]] = []
    path_list: List[slice] = []
    for s, e in zip(start, end):
        if s == e:
            path_list.append(slice(s, s + 1))
        else:
            break
    path: Tuple[slice, ...] = tuple(path_list)
    divergence_idx = len(path)
    if divergence_idx == len(dims):
        return [path]

    def upper() -> Tuple[Tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None
        sdi = start[divergence_idx]
        return tuple((path + (slice(sdi, sdi + 1),) + s for s in _get_minimal_slice_set(start[divergence_idx + 1:], [d - 1 for d in dims[divergence_idx + 1:]], dims[divergence_idx + 1:], start_edges=start_edges[divergence_idx + 1:], end_edges=[True for _ in end_edges[divergence_idx + 1:]])))

    def lower() -> Tuple[Tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None
        edi = end[divergence_idx]
        return tuple((path + (slice(edi, edi + 1),) + s for s in _get_minimal_slice_set([0 for _ in start[divergence_idx + 1:]], end[divergence_idx + 1:], dims[divergence_idx + 1:], start_edges=[True for _ in start_edges[divergence_idx + 1:]], end_edges=end_edges[divergence_idx + 1:])))
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())
    return slices