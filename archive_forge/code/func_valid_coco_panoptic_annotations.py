import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def valid_coco_panoptic_annotations(annotations: Iterable[Dict[str, Union[List, Tuple]]]) -> bool:
    return all((is_valid_annotation_coco_panoptic(ann) for ann in annotations))