import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    gurantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self):
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0) # In-place addition

                return output

    Resulting Graph would be::

        graph():
            %arg0_1 := placeholder[target=arg0_1]
            %arg1_1 := placeholder[target=arg1_1]
            %arg2_1 := placeholder[target=arg2_1]
            %arg3_1 := placeholder[target=arg3_1]
            %arg4_1 := placeholder[target=arg4_1]
            %add_tensor := call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %arg0_1), kwargs = {})
            %mul_tensor := call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %arg1_1), kwargs = {})
            %mul_tensor_1 := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %arg2_1), kwargs = {})
            %add_tensor_1 := call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
            %add_tensor_2 := call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1.0), kwargs = {})
            return (add_tensor_2, add_tensor_1)

    Resulting ExportGraphSignature would be::

        ExportGraphSignature(
            input_specs=[
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='my_parameter'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg1_1'), target='my_buffer1'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg2_1'), target='my_buffer2'),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg4_1'), target=None)
            ],
            output_specs=[
                OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add_2'), target='my_buffer2'),
                OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add_1'), target=None)
            ]
        )
    """
    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]

    @property
    def parameters(self) -> Collection[str]:
        return [s.target for s in self.input_specs if s.kind == InputKind.PARAMETER if isinstance(s.target, str)]

    @property
    def buffers(self) -> Collection[str]:
        return [s.target for s in self.input_specs if s.kind == InputKind.BUFFER if isinstance(s.target, str)]

    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        return [s.target for s in self.input_specs if s.kind == InputKind.CONSTANT_TENSOR if isinstance(s.target, str)]

    @property
    def user_inputs(self) -> Collection[str]:
        return tuple((s.arg.name for s in self.input_specs if s.kind == InputKind.USER_INPUT and isinstance(s.arg, TensorArgument)))

    @property
    def user_outputs(self) -> Collection[str]:
        return tuple((s.arg.name for s in self.output_specs if s.kind == OutputKind.USER_OUTPUT and isinstance(s.arg, TensorArgument)))

    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.PARAMETER and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.BUFFER and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        return {s.arg.name: s.target for s in self.output_specs if s.kind == OutputKind.BUFFER_MUTATION and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]:
        return {s.arg.name: s.target for s in self.output_specs if s.kind == OutputKind.USER_INPUT_MUTATION and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.CONSTANT_TENSOR and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}

    @property
    def backward_signature(self) -> Optional[ExportBackwardSignature]:
        loss_output = None
        gradients_to_parameters: Dict[str, str] = {}
        gradients_to_user_inputs: Dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                assert loss_output is None
                assert isinstance(spec.arg, TensorArgument)
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_user_inputs[spec.arg.name] = spec.target
        if loss_output is None:
            return None
        return ExportBackwardSignature(loss_output=loss_output, gradients_to_parameters=gradients_to_parameters, gradients_to_user_inputs=gradients_to_user_inputs)

    @property
    def assertion_dep_token(self) -> Optional[Mapping[int, str]]:
        return None

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token) == 1
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        assert len(self.user_outputs) + len(self.buffers_to_mutate) == assertion_dep_token_index

    def replace_all_uses(self, old: str, new: str):
        """
        Replace all uses of the old name with new name in the signature.
        """
        assert isinstance(old, str)
        assert isinstance(new, str)
        for o in self.output_specs:
            if isinstance(o.arg, TensorArgument):
                if o.arg.name == old:
                    o.arg.name = new