from typing import List, Union
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
def _parse_cpu_cores(cpu_cores: Union[int, str, List[int]]) -> int:
    """Parses the cpu_cores given in the format as accepted by the ``devices`` argument in the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`.

    Args:
        cpu_cores: An int > 0.

    Returns:
        An int representing the number of processes

    Raises:
        MisconfigurationException:
            If cpu_cores is not an int > 0

    """
    if isinstance(cpu_cores, str) and cpu_cores.strip().isdigit():
        cpu_cores = int(cpu_cores)
    if not isinstance(cpu_cores, int) or cpu_cores <= 0:
        raise TypeError('`devices` selected with `CPUAccelerator` should be an int > 0.')
    return cpu_cores