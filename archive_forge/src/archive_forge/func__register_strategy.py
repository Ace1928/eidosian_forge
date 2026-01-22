import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def _register_strategy(strategy: Callable, name: str):

    @wraps(strategy)
    def new_func(old_state: ReproState, granularity=1):
        print(file=sys.stderr)
        print(f'Strategy: {name} (G: {granularity}) ({len(old_state.graph.nodes)} nodes, {len(old_state.inps)} inputs)', file=sys.stderr)
        new_state = strategy(deepcopy_fx_graph(old_state.graph), list(old_state.inps), granularity)
        if new_state is not None:
            new_nodes = len(new_state.graph.nodes)
            old_nodes = len(old_state.graph.nodes)
            new_inps = len(new_state.inps)
            old_inps = len(old_state.inps)
            new_outs = len(get_outputs(new_state.graph))
            old_outs = len(get_outputs(old_state.graph))
            progress_made = False
            if new_nodes < old_nodes:
                progress_made = True
                print(f'SUCCESS: Went from {old_nodes} to {new_nodes} nodes', file=sys.stderr)
            if new_inps > old_inps:
                progress_made = True
                print(f'SUCCESS: Went from {old_inps} to {new_inps} inputs', file=sys.stderr)
            if new_outs < old_outs:
                progress_made = True
                print(f'SUCCESS: Went from {old_outs} to {new_outs} outputs', file=sys.stderr)
            if not progress_made:
                raise RuntimeError('Success raised but no progress made?')
            if not graph_fails(new_state.graph, new_state.inps):
                print('WARNING: Something went wrong, not applying this minification', file=sys.stderr)
                return None
            return new_state
        else:
            print(f'FAIL: {name}', file=sys.stderr)
        return None
    return new_func