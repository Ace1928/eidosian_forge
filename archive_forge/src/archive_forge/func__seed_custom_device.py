import contextlib
from typing import Generator
import warnings
from torch._C import default_generator
import torch
def _seed_custom_device(seed) -> None:
    """Sets the seed to generate random numbers for custom device.

    Args:
        seed (int): The desired seed.

    See [Note: support the custom device with privateuse1]
    """
    seed = int(seed)
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    if hasattr(torch, custom_backend_name):
        custom_device_mod = getattr(torch, custom_backend_name)
        _bad_fork_name = '_is_in_bad_fork'
        _seed_all_name = 'manual_seed_all'
        if hasattr(custom_device_mod, _bad_fork_name) and hasattr(custom_device_mod, _seed_all_name):
            if not getattr(custom_device_mod, _bad_fork_name)():
                getattr(custom_device_mod, _seed_all_name)(seed)
        else:
            message = f"Set seed for `{custom_backend_name}` device does not take effect, please add API's "
            message += f'`{_bad_fork_name}` and `{_seed_all_name}` to `{custom_backend_name}` device module.'
            warnings.warn(message, UserWarning, stacklevel=3)