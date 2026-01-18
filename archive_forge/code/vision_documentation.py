import os
from typing import Any, Callable, List, Optional, Tuple
import torch.utils.data as data
from ..utils import _log_api_usage_once

        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        