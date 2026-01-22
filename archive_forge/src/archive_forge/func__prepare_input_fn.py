from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
def _prepare_input_fn(self, inputs, device_mesh):
    prepared_inputs = []
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    for inp, input_layout, desired_layout in zip(inputs, self.input_layouts, self.desired_input_layouts):
        if input_layout is not None:
            if isinstance(inp, DTensor):
                assert inp.placements[0] == input_layout
                dt_inp = inp
            else:
                dt_inp = DTensor.from_local(inp, device_mesh, (input_layout,), run_check=False)
            if input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))
            prepared_inputs.append(dt_inp.to_local() if self.use_local_output else dt_inp)
        else:
            prepared_inputs.append(inp)
    return tuple(prepared_inputs)