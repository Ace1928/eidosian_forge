from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from huggingface_hub.utils import parse_datetime
class SpaceHardware(str, Enum):
    """
    Enumeration of hardwares available to run your Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceHardware.CPU_BASIC == "cpu-basic"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceInfo.ts#L73 (private url).
    """
    CPU_BASIC = 'cpu-basic'
    CPU_UPGRADE = 'cpu-upgrade'
    T4_SMALL = 't4-small'
    T4_MEDIUM = 't4-medium'
    ZERO_A10G = 'zero-a10g'
    A10G_SMALL = 'a10g-small'
    A10G_LARGE = 'a10g-large'
    A10G_LARGEX2 = 'a10g-largex2'
    A10G_LARGEX4 = 'a10g-largex4'
    A100_LARGE = 'a100-large'