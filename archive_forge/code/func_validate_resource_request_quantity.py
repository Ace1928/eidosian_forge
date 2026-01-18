import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def validate_resource_request_quantity(quantity: float) -> Tuple[bool, Optional[str]]:
    if quantity not in TPU_VALID_CHIP_OPTIONS:
        return (False, f"The number of requested 'TPU' was set to {quantity} which is not a supported chip configuration. Supported configs: {TPU_VALID_CHIP_OPTIONS}")
    else:
        return (True, None)