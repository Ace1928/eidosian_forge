from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder.
        desired_input_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> ...
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(self, *, input_layouts: Union[Placement, Tuple[Placement]], desired_input_layouts: Union[Placement, Tuple[Placement]], use_local_output: bool=False):
        self.input_layouts = (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        self.desired_input_layouts = (desired_input_layouts,) if isinstance(desired_input_layouts, Placement) else desired_input_layouts
        self.use_local_output = use_local_output
        assert len(self.input_layouts) == len(self.desired_input_layouts), 'input_layouts and desired_input_layouts should have same length!'

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

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_pre_hook(lambda _, inputs: self._prepare_input_fn(inputs, device_mesh))
        return module