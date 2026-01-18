import copy
from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple
import warnings
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRanges
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
from torch.export.graph_signature import (
from torch.export.exported_program import (
from .utils import _check_input_constraints_pre_hook
def unlift_exported_program_lifted_states(ep: torch.export.ExportedProgram) -> torch.nn.Module:
    new_gm = copy.deepcopy(ep.graph_module)
    inp_pos_to_param_buffer_name = _construct_inp_pos_to_param_buffer_name(new_gm, ep.graph_signature, ep.state_dict, ep.tensor_constants)
    new_gm = _unlift(new_gm, inp_pos_to_param_buffer_name, ep.call_spec.in_spec, ep.call_spec.out_spec, ep.state_dict, ep.tensor_constants, ep.graph_signature.buffers_to_mutate)
    unlift_gm = _create_stateful_graph_module(new_gm, ep.range_constraints, ep.equality_constraints)
    unlift_gm.meta.update(ep.graph_module.meta)
    return unlift_gm