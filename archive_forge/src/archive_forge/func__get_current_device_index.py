import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def _get_current_device_index():
    _get_device_index = 'current_device'
    if hasattr(torch, custom_backend_name) and hasattr(getattr(torch, custom_backend_name), _get_device_index):
        return getattr(getattr(torch, custom_backend_name), _get_device_index)()
    else:
        return 0