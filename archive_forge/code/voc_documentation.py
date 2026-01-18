import collections
import os
from xml.etree.ElementTree import Element as ET_Element
from .vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        