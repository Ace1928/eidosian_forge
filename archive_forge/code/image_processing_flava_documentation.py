import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging

        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_image_mask (`bool`, *optional*, defaults to `self.return_image_mask`):
                Whether to return the image mask.
            input_size_patches (`int`, *optional*, defaults to `self.input_size_patches`):
                Size of the patches to extract from the image.
            total_mask_patches (`int`, *optional*, defaults to `self.total_mask_patches`):
                Total number of patches to extract from the image.
            mask_group_min_patches (`int`, *optional*, defaults to `self.mask_group_min_patches`):
                Minimum number of patches to extract from the image.
            mask_group_max_patches (`int`, *optional*, defaults to `self.mask_group_max_patches`):
                Maximum number of patches to extract from the image.
            mask_group_min_aspect_ratio (`float`, *optional*, defaults to `self.mask_group_min_aspect_ratio`):
                Minimum aspect ratio of the patches to extract from the image.
            mask_group_max_aspect_ratio (`float`, *optional*, defaults to `self.mask_group_max_aspect_ratio`):
                Maximum aspect ratio of the patches to extract from the image.
            return_codebook_pixels (`bool`, *optional*, defaults to `self.return_codebook_pixels`):
                Whether to return the codebook pixels.
            codebook_do_resize (`bool`, *optional*, defaults to `self.codebook_do_resize`):
                Whether to resize the codebook pixels.
            codebook_size (`Dict[str, int]`, *optional*, defaults to `self.codebook_size`):
                Size of the codebook pixels.
            codebook_resample (`int`, *optional*, defaults to `self.codebook_resample`):
                Resampling filter to use if resizing the codebook pixels. This can be one of the enum
                `PILImageResampling`, Only has an effect if `codebook_do_resize` is set to `True`.
            codebook_do_center_crop (`bool`, *optional*, defaults to `self.codebook_do_center_crop`):
                Whether to center crop the codebook pixels.
            codebook_crop_size (`Dict[str, int]`, *optional*, defaults to `self.codebook_crop_size`):
                Size of the center crop of the codebook pixels. Only has an effect if `codebook_do_center_crop` is set
                to `True`.
            codebook_do_rescale (`bool`, *optional*, defaults to `self.codebook_do_rescale`):
                Whether to rescale the codebook pixels values between [0 - 1].
            codebook_rescale_factor (`float`, *optional*, defaults to `self.codebook_rescale_factor`):
                Rescale factor to rescale the codebook pixels by if `codebook_do_rescale` is set to `True`.
            codebook_do_map_pixels (`bool`, *optional*, defaults to `self.codebook_do_map_pixels`):
                Whether to map the codebook pixels values.
            codebook_do_normalize (`bool`, *optional*, defaults to `self.codebook_do_normalize`):
                Whether to normalize the codebook pixels.
            codebook_image_mean (`float` or `List[float]`, *optional*, defaults to `self.codebook_image_mean`):
                Codebook pixels mean to normalize the codebook pixels by if `codebook_do_normalize` is set to `True`.
            codebook_image_std (`float` or `List[float]`, *optional*, defaults to `self.codebook_image_std`):
                Codebook pixels standard deviation to normalize the codebook pixels by if `codebook_do_normalize` is
                set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        