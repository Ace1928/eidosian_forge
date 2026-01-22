import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
class AnnotionFormat(ExplicitEnum):
    COCO_DETECTION = AnnotationFormat.COCO_DETECTION.value
    COCO_PANOPTIC = AnnotationFormat.COCO_PANOPTIC.value