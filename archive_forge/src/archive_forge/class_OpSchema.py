from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it
    includes DTensor DTensorSpecs and non-tensor args/kwargs (positional order
    preserved). It is mainly used by the dispatching logic below to run things like
    sharding propagation.

    NOTE: this should be used as a read only data class
    TODO: make this a frozen dataclass

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec
    """
    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType
    schema_info: Optional[RuntimeSchemaInfo] = None

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        return tuple((item for item in self.args_schema if isinstance(item, DTensorSpec)))

    def __repr__(self) -> str:
        return f'OpSchema(op={self.op}, args_schema={self.args_schema}, kwargs_schema={self.kwargs_schema})'

    def __str__(self) -> str:
        args_sharding: List[str] = []
        mesh_shape = None
        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_sharding.append(str(arg))
                mesh_shape = arg.mesh.shape
            elif isinstance(arg, OpStrategy):
                assert len(arg.strategies) == 1
                arg_spec = arg.strategies[0].output_spec
                args_sharding.append(str(arg_spec))
                mesh_shape = arg_spec.mesh.shape
            elif isinstance(arg, TupleStrategy):
                first_op_strtgy = arg.childs[0]
                assert isinstance(first_op_strtgy, OpStrategy)
                mesh_shape = first_op_strtgy.strategies[0].output_spec.mesh.shape
                args_sharding.append(str(arg))
            else:
                args_sharding.append(str(arg))
        return f'Op(op={self.op}, args_sharding={', '.join(args_sharding)}@ mesh: {mesh_shape})'

    def __post_init__(self) -> None:
        has_symints = False
        for a in self.args_schema:
            if isinstance(a, DTensorSpec) and a.tensor_meta is not None:
                if any((isinstance(s, torch.SymInt) for s in a.tensor_meta.shape)):
                    has_symints = True
                    break
        self.has_symints = has_symints

    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        arg = self.args_schema[arg_idx]
        is_tensor = isinstance(arg, DTensorSpec)
        if is_tensor:
            return True
        if not isinstance(arg, list):
            return False
        return all((isinstance(e, DTensorSpec) or e is None for e in arg))

    def return_type_tuple_tensors(self) -> bool:
        return_types = self.op._schema.returns
        return len(return_types) > 1 and isinstance(return_types[0].type, torch.TensorType)

    def return_type_tensor(self) -> bool:
        return_types = self.op._schema.returns
        return isinstance(return_types[0].type, torch.TensorType)

    def __hash__(self) -> int:
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey
        args_to_hash = tuple((tuple(e) if isinstance(e, list) else e for i, e in enumerate(self.args_schema) if self.arg_type_tensor_or_tensor_list_like(i) or i >= static_argnum))
        if static_kwargkey is not None:
            kwargs_to_hash = tuple((self.kwargs_schema.get(k, None) for k in static_kwargkey))
            return hash((self.op, args_to_hash, kwargs_to_hash))
        else:
            return hash((self.op, args_to_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSchema):
            return False
        if self.op != other.op:
            return False
        if len(self.args_schema) != len(other.args_schema):
            return False
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey
        for i, (self_arg, other_arg) in enumerate(zip(self.args_schema, other.args_schema)):
            if isinstance(self_arg, DTensorSpec) and self_arg != other_arg:
                return False
            elif i >= static_argnum and self_arg != other_arg:
                return False
        if static_kwargkey:
            for key in static_kwargkey:
                if self.kwargs_schema.get(key, None) != other.kwargs_schema.get(key, None):
                    return False
        return True

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.args_schema)

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map_only(DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.kwargs_schema)

    def _inplace_rewrap_schema_suggestion(self, origin_schema: 'OpSchema') -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: List[object] = []
        idx_of_args_spec = 0
        for arg in origin_schema.args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema