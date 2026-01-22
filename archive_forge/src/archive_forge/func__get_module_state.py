from typing import cast, Dict, Optional
import torch.nn as nn
def _get_module_state(module: nn.Module) -> Optional[_State]:
    """
    Return the ``_State`` in ``model``.

    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """
    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    elif module in _module_state_mapping:
        return _module_state_mapping[module]
    else:
        return None