from __future__ import annotations
import os
from typing import Any, Callable, Optional, Tuple
import PIL.Image
from .utils import download_and_extract_archive, verify_str_arg
from .vision import VisionDataset

        Download the FGVC Aircraft dataset archive and extract it under root.
        