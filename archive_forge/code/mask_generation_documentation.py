from collections import defaultdict
from typing import Optional
from ..image_utils import load_image
from ..utils import (
from .base import ChunkPipeline, build_pipeline_init_args

        Generates binary segmentation masks

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                Image or list of images.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold to use when turning the predicted masks into binary values.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                A filtering threshold in `[0,1]` applied on the model's predicted mask quality.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                A filtering threshold in `[0,1]`, using the stability of the mask under changes to the cutoff used to
                binarize the model's mask predictions.
            stability_score_offset (`int`, *optional*, defaults to 1):
                The amount to shift the cutoff when calculated the stability score.
            crops_nms_thresh (`float`, *optional*, defaults to 0.7):
                The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
            crops_n_layers (`int`, *optional*, defaults to 0):
                If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of
                layers to run, where each layer has 2**i_layer number of image crops.
            crop_overlap_ratio (`float`, *optional*, defaults to `512 / 1500`):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            crop_n_points_downscale_factor (`int`, *optional*, defaults to `1`):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            `Dict`: A dictionary with the following keys:
                - **mask** (`PIL.Image`) -- A binary mask of the detected object as a PIL Image of shape `(width,
                  height)` of the original image. Returns a mask filled with zeros if no object is found.
                - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of
                  the "object" described by the label and the mask.

        