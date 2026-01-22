import os
import shutil
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
def _get_segmentation_target(self, filepath: str) -> Image.Image:
    mat = self._loadmat(filepath)
    return Image.fromarray(mat['GTcls'][0]['Segmentation'][0])