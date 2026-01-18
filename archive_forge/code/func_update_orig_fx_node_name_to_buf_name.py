import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
def update_orig_fx_node_name_to_buf_name(nodes: SchedulerNodeList, node_name_to_buf_name: Dict[str, str], parent_buf_name: Optional[str]=None, n_origins: int=0):
    if nodes is None:
        return
    for node in nodes:
        buf_name = node.get_name()
        children_nodes = node.get_nodes()
        if children_nodes is not None and len(children_nodes) > 1:
            update_orig_fx_node_name_to_buf_name(children_nodes, node_name_to_buf_name, buf_name if parent_buf_name is None else parent_buf_name)
            continue
        else:
            assert len(children_nodes) == 1 and children_nodes[0] == node
        ir_node = node.node
        if ir_node is None or ir_node.origins is None:
            continue
        for origin in ir_node.origins:
            node_name = origin.name
            if node_name not in node_name_to_buf_name:
                node_name_to_buf_name[node_name] = buf_name if parent_buf_name is None else parent_buf_name