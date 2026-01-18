import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is `0` for different indentities and `1` for same identities.
        