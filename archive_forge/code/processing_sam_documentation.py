from copy import deepcopy
from typing import Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_tf_available, is_torch_available

        Check and preprocesses the 2D points, labels and bounding boxes. It checks if the input is valid and if they
        are, it converts the coordinates of the points and bounding boxes. If a user passes directly a `torch.Tensor`,
        it is converted to a `numpy.ndarray` and then to a `list`.
        