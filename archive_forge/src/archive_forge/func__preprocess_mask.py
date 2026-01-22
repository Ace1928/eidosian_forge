import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _preprocess_mask(self, segmentation_map: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
    """Preprocesses a single mask."""
    segmentation_map = to_numpy_array(segmentation_map)
    if segmentation_map.ndim == 2:
        added_channel_dim = True
        segmentation_map = segmentation_map[None, ...]
        input_data_format = ChannelDimension.FIRST
    else:
        added_channel_dim = False
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
    segmentation_map = self._preprocess(image=segmentation_map, do_resize=do_resize, resample=PILImageResampling.NEAREST, size=size, do_rescale=False, do_normalize=False, input_data_format=input_data_format)
    if added_channel_dim:
        segmentation_map = segmentation_map.squeeze(0)
    return segmentation_map