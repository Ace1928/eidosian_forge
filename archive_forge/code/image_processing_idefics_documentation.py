from typing import Callable, Dict, List, Optional, Union
from PIL import Image
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available

        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Resize to image size
            image_num_channels (`int`, *optional*, defaults to `self.image_num_channels`):
                Number of image channels.
            image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
                Mean to use if normalizing the image. This is a float or list of floats the length of the number of
                channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
                be overridden by the `image_mean` parameter in the `preprocess` method.
            image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
                Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
                number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
                method. Can be overridden by the `image_std` parameter in the `preprocess` method.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images

        Returns:
            a PyTorch tensor of the processed images

        