import os
import glob
import logging
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
Get the type of the Ascend NPU on the current node.

        Returns:
            A string of the type, such as "Ascend910A", "Ascend910B", "Ascend310P1".
        