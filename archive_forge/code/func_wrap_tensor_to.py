import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def wrap_tensor_to(self: torch.Tensor, device: Optional[Union[int, torch.device]]=None, non_blocking=False, **kwargs) -> torch.Tensor:
    """Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
    device_idx = _normalization_device(custom_backend_name, device)
    return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)