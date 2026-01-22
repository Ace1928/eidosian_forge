from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from huggingface_hub.utils import parse_datetime
class SpaceStorage(str, Enum):
    """
    Enumeration of persistent storage available for your Space on the Hub.

    Value can be compared to a string:
    ```py
    assert SpaceStorage.SMALL == "small"
    ```

    Taken from https://github.com/huggingface/moon-landing/blob/main/server/repo_types/SpaceHardwareFlavor.ts#L24 (private url).
    """
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'