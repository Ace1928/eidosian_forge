import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def wrap_module_to(self: torch.nn.modules.module.T, device: Optional[Union[int, torch.device]]=None) -> torch.nn.modules.module.T:
    """Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
    return self._apply(lambda t: getattr(t, custom_backend_name)(device))