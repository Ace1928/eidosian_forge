import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _to(elem):
    if torch.is_floating_point(elem):
        return elem.to(*args, **kwargs)
    if device is not None:
        return elem.to(device=device)
    return elem