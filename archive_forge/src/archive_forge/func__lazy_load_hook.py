import itertools
import warnings
from typing import Protocol
import torch
from ..parameter import is_lazy
def _lazy_load_hook(self: _LazyProtocol, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
    for name, param in itertools.chain(self._parameters.items(), self._buffers.items()):
        key = prefix + name
        if key in state_dict and param is not None:
            input_param = state_dict[key]
            if is_lazy(param):
                if not is_lazy(input_param):
                    with torch.no_grad():
                        param.materialize(input_param.shape)