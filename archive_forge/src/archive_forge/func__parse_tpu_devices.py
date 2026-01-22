import functools
from typing import Any, List, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _check_data_type
def _parse_tpu_devices(devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
    """Parses the TPU devices given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.trainer.Trainer` and :class:`~lightning_fabric.Fabric`.

    Args:
        devices: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A single element list of int or string can be used to indicate the specific TPU core to use.

    Returns:
        A list of tpu cores to be used.

    """
    _check_data_type(devices)
    if isinstance(devices, str):
        devices = _parse_tpu_devices_str(devices)
    _check_tpu_devices_valid(devices)
    return devices