import inspect
from collections.abc import Iterable
from typing import Optional, Text
def _signature_validation(self, qnode, weight_shapes):
    sig = inspect.signature(qnode.func).parameters
    if self.input_arg not in sig:
        raise TypeError(f'QNode must include an argument with name {self.input_arg} for inputting data')
    if self.input_arg in set(weight_shapes.keys()):
        raise ValueError(f'{self.input_arg} argument should not have its dimension specified in weight_shapes')
    param_kinds = [p.kind for p in sig.values()]
    if inspect.Parameter.VAR_POSITIONAL in param_kinds:
        raise TypeError('Cannot have a variable number of positional arguments')
    if inspect.Parameter.VAR_KEYWORD not in param_kinds:
        if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
            raise ValueError('Must specify a shape for every non-input parameter in the QNode')