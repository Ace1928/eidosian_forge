import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import extract_archive, iterable_to_str, verify_str_arg
from .vision import VisionDataset
def _get_target_suffix(self, mode: str, target_type: str) -> str:
    if target_type == 'instance':
        return f'{mode}_instanceIds.png'
    elif target_type == 'semantic':
        return f'{mode}_labelIds.png'
    elif target_type == 'color':
        return f'{mode}_color.png'
    else:
        return f'{mode}_polygons.json'