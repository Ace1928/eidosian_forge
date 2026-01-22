from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from functools import partial
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor, distribute_module
class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> ...
        >>> # By default, the input of the "w1" Linear will be annotated to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>>
        >>> parallelize_module(
        >>>     module=block, # this can be a submodule or module
        >>>     ...,
        >>>     parallelize_plan={"w1": ColwiseParallel()},
        >>> )
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    def __init__(self, *, input_layouts: Optional[Placement]=None, output_layouts: Optional[Placement]=None, use_local_output: bool=True):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts)
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        if isinstance(module, nn.Linear):
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
                module.register_parameter(name, dist_param)
        elif isinstance(module, nn.Embedding):
            for name, param in module.named_parameters():
                dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(1)]))
                module.register_parameter(name, dist_param)
        else:
            raise NotImplementedError(f'ColwiseParallel only supports nn.Linearand nn.Embedding for now, but found {type(module)}!')

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, outputs, device_mesh):
        outputs = outputs.redistribute(placements=output_layouts)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, self._partition_fn, partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts), partial(self._prepare_output_fn, self.output_layouts, self.use_local_output))