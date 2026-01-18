import os
from os.path import abspath, expanduser
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from .utils import download_and_extract_archive, download_file_from_google_drive, extract_archive, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dict of annotations for all faces in the image.
            target=None for the test split.
        