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
def _preprocess(self, image: ImageInput, do_resize: bool=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_rescale: bool=None, rescale_factor: float=None, do_normalize: bool=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None):
    if do_resize:
        image = self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
    if do_rescale:
        image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
    if do_normalize:
        image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
    return image