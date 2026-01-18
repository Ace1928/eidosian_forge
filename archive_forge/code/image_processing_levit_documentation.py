from typing import Dict, Iterable, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging

        Preprocess an image or batch of images to be used as input to a LeViT model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image
                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value
                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value
                i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the output image after center cropping. Crops images to (crop_size["height"],
                crop_size["width"]).
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values by `rescaling_factor` - typical to values between 0 and 1.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Factor to rescale the image pixel values by.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image pixel values by `image_mean` and `image_std`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to normalize the image pixel values by.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to normalize the image pixel values by.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        