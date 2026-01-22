import os
from typing import Dict, List
import pyarrow.fs
from ray.tune.logger import LoggerCallback
from ray.tune.experiment import Trial
from ray.tune.utils import flatten_dict
def _check_key_name(self, key: str, item: str) -> bool:
    """
        Check if key argument is equal to item argument or starts with item and
        a forward slash. Used for parsing trial result dictionary into ignored
        keys, system metrics, episode logs, etc.
        """
    return key.startswith(item + '/') or key == item