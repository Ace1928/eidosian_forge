import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        