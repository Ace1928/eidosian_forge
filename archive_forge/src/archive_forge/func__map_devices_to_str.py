import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Dict, Optional
import numpy as np
import pyarrow as pa
from .. import config
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter
@staticmethod
def _map_devices_to_str() -> Dict[str, 'jaxlib.xla_extension.Device']:
    import jax
    return {str(device): device for device in jax.devices()}