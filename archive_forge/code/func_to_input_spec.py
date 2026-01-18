import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
def to_input_spec(i: ArgumentSpec) -> InputSpec:
    if not isinstance(i, TensorArgument):
        return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
    name = i.name
    if name in user_inputs:
        return InputSpec(kind=InputKind.USER_INPUT, arg=i, target=None)
    elif name in inputs_to_parameters:
        return InputSpec(kind=InputKind.PARAMETER, arg=i, target=inputs_to_parameters[name])
    elif name in inputs_to_buffers:
        return InputSpec(kind=InputKind.BUFFER, arg=i, target=inputs_to_buffers[name])
    else:
        raise AssertionError(f'Unknown tensor input kind: {name}')