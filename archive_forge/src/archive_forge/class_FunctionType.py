import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
class FunctionType(inspect.Signature):
    """Represents the type of a TensorFlow function.

  FunctionType is the canonical way to represent the input/output contract of
  all kinds of functions within the tf.function domain, including:
    - Polymorphic Function
    - Concrete Function
    - Atomic Function

  It provides consistent, centralized and layered logic for:
    - Canonicalization of Python input arguments
    - Type-based dispatch to monomorphic functions
    - Packing/unpacking structured python values to Tensors
    - Generation of structured placeholder values for tracing

  Additionaly, it also provides:
    - Lossless serialization
    - Native integration with Python function signature representation
    - Seamless migration from older representation formats
  """

    def __init__(self, parameters: Sequence[inspect.Parameter], captures: Optional[collections.OrderedDict]=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._captures = captures if captures else collections.OrderedDict()

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Returns an ordered mapping of parameter name to specification."""
        return super().parameters

    @property
    def captures(self) -> collections.OrderedDict:
        """Returns an ordered mapping of capture id to type."""
        return self._captures

    @property
    def output(self) -> Optional[trace.TraceType]:
        """Return the output TraceType if specified."""
        return self.return_annotation if self.return_annotation is not self.empty else None

    @classmethod
    def from_callable(cls, obj: Callable[..., Any], *, follow_wrapped: bool=True) -> 'FunctionType':
        """Generate FunctionType from a python Callable."""
        signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
        parameters = [Parameter(p.name, p.kind, p.default is not p.empty, None) for p in signature.parameters.values()]
        return FunctionType(parameters)

    @classmethod
    def get_default_values(cls, obj: Callable[..., Any], *, follow_wrapped: bool=True) -> Dict[str, Any]:
        """Inspects and returns a dictionary of default values."""
        signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
        default_values = {}
        for p in signature.parameters.values():
            if p.default is not p.empty:
                default_values[p.name] = p.default
        return default_values

    @classmethod
    def from_proto(cls, proto: Any) -> 'FunctionType':
        """Generate a FunctionType from the proto representation."""
        return FunctionType([Parameter.from_proto(p) for p in proto.parameters], collections.OrderedDict([(c.name, serialization.deserialize(c.type_constraint)) for c in proto.captures]))

    def to_proto(self) -> Any:
        """Generate a proto representation from the FunctionType."""
        return function_type_pb2.FunctionType(parameters=[p.to_proto() for p in self.parameters.values()], captures=[function_type_pb2.Capture(name=n, type_constraint=serialization.serialize(t)) for n, t in self.captures.items()])

    def bind_with_defaults(self, args, kwargs, default_values):
        """Returns BoundArguments with default values filled in."""
        bound_arguments = self.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        with_default_args = collections.OrderedDict()
        for name, value in bound_arguments.arguments.items():
            if value is CAPTURED_DEFAULT_VALUE:
                with_default_args[name] = default_values[name]
            else:
                with_default_args[name] = value
        for arg_name in with_default_args:
            constraint = self.parameters[arg_name].type_constraint
            if constraint:
                with_default_args[arg_name] = constraint._cast(with_default_args[arg_name], trace_type.InternalCastContext(allow_specs=True))
        bound_arguments = inspect.BoundArguments(self, with_default_args)
        return bound_arguments

    def is_supertype_of(self, other: 'FunctionType') -> bool:
        """Returns True if self is a supertype of other FunctionType."""
        if len(self.parameters) != len(other.parameters):
            return False
        for self_param, other_param in zip(self.parameters.values(), other.parameters.values()):
            if not self_param.is_subtype_of(other_param):
                return False
        if not all((name in other.captures for name in self.captures)):
            return False
        return all((capture_type.is_subtype_of(other.captures[name]) for name, capture_type in self.captures.items()))

    def most_specific_common_subtype(self, others: Sequence['FunctionType']) -> Optional['FunctionType']:
        """Returns a common subtype (if exists)."""
        subtyped_parameters = []
        for i, parameter in enumerate(self.parameters.values()):
            subtyped_parameter = parameter.most_specific_common_supertype([list(other.parameters.values())[i] for other in others])
            if subtyped_parameter is None:
                return None
            subtyped_parameters.append(subtyped_parameter)
        if not all(subtyped_parameters):
            return None
        capture_names = set(self.captures.keys())
        for other in others:
            capture_names = capture_names.union(other.captures.keys())
        subtyped_captures = collections.OrderedDict()
        for name in capture_names:
            containing = [t for t in [self, *others] if name in t.captures]
            base = containing[0]
            relevant_others = containing[1:]
            common_type = base.captures[name].most_specific_common_supertype([other.captures[name] for other in relevant_others])
            if common_type is None:
                return None
            else:
                subtyped_captures[name] = common_type
        return FunctionType(subtyped_parameters, subtyped_captures)

    def placeholder_arguments(self, placeholder_context: trace.PlaceholderContext) -> inspect.BoundArguments:
        """Returns BoundArguments of values that can be used for tracing."""
        arguments = collections.OrderedDict()
        for parameter in self.parameters.values():
            if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
                raise ValueError('Can not generate placeholder values for variable length function type.')
            if not parameter.type_constraint:
                raise ValueError('Can not generate placeholder value for partially defined function type.')
            placeholder_context.update_naming_scope(parameter.name)
            arguments[parameter.name] = parameter.type_constraint.placeholder_value(placeholder_context)
        return inspect.BoundArguments(self, arguments)

    @property
    def flat_inputs(self) -> List[trace.TraceType]:
        """Flat tensor inputs accepted by this FunctionType."""
        if not hasattr(self, '_cached_flat_inputs'):
            cached_flat_inputs = []
            for p in self.parameters.values():
                cached_flat_inputs.extend(p.type_constraint._flatten())
            self._cached_flat_inputs = cached_flat_inputs
        return self._cached_flat_inputs

    def unpack_inputs(self, bound_parameters: inspect.BoundArguments) -> List[core.Tensor]:
        """Unpacks python arguments to flat tensor inputs accepted by this type."""
        sorted_parameters = []
        kwonly_parameters = []
        for p in self.parameters.values():
            if p.kind is Parameter.KEYWORD_ONLY:
                kwonly_parameters.append(p)
            else:
                sorted_parameters.append(p)
        sorted_parameters = sorted_parameters + sorted(kwonly_parameters, key=lambda p: p.name)
        flat = []
        for p in sorted_parameters:
            flat.extend(p.type_constraint._to_tensors(bound_parameters.arguments[p.name]))
        dealiased_inputs = []
        ids_used = set()
        for tensor, input_type in zip(flat, self.flat_inputs):
            alias_id = input_type._alias_id()
            if alias_id is None or alias_id not in ids_used:
                dealiased_inputs.append(tensor)
            if alias_id is not None:
                ids_used.add(alias_id)
        return dealiased_inputs

    @property
    def flat_captures(self) -> List[trace.TraceType]:
        """Flat tensor captures needed by this FunctionType."""
        if not hasattr(self, '_cached_flat_captures'):
            cached_flat_captures = []
            for t in self.captures.values():
                cached_flat_captures.extend(t._flatten())
            self._cached_flat_captures = cached_flat_captures
        return self._cached_flat_captures

    def unpack_captures(self, captures) -> List[core.Tensor]:
        """Unpacks captures to flat tensors."""
        flat = []
        for v, t in zip(captures, self.captures.values()):
            flat.extend(t._to_tensors(v))
        if len(flat) != len(self.flat_captures):
            raise TypeError(f'Flattening captures {captures} with type {self!r} produced {len(flat)} tensors instead of {len(self.flat_captures)}')
        return flat

    @property
    def flat_outputs(self) -> List[trace.TraceType]:
        """Flat tensor outputs returned by this FunctionType."""
        if not hasattr(self, '_cached_flat_outputs'):
            if self.output is not None:
                self._cached_flat_outputs = self.output._flatten()
        return self._cached_flat_outputs

    def pack_output(self, flat_values: Sequence[core.Tensor]) -> Any:
        """Packs flat tensors to generate a value of the output type."""
        if flat_values is None:
            flat_values = []
        if self.output is None:
            raise ValueError('Can not pack outputs for undefined output type.')
        else:
            return self.output._from_tensors(iter(flat_values))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FunctionType):
            return NotImplemented
        return (self.parameters, self.captures) == (other.parameters, other.captures)

    def __hash__(self) -> int:
        return hash((tuple(self.parameters.items()), tuple(self.captures.items())))

    def __repr__(self):
        return f'FunctionType(parameters={list(self.parameters.values())!r}, captures={self.captures})'