import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import check_integrity, download_url, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        