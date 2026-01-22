import functools
import numpy as np
from .. import color
from ..util.dtype import _convert
def adapt_rgb(apply_to_rgb):
    """Return decorator that adapts to RGB images to a gray-scale filter.

    This function is only intended to be used for functions that don't accept
    volumes as input, since checking an image's shape is fragile.

    Parameters
    ----------
    apply_to_rgb : function
        Function that returns a filtered image from an image-filter and RGB
        image. This will only be called if the image is RGB-like.
    """

    def decorator(image_filter):

        @functools.wraps(image_filter)
        def image_filter_adapted(image, *args, **kwargs):
            if is_rgb_like(image):
                return apply_to_rgb(image_filter, image, *args, **kwargs)
            else:
                return image_filter(image, *args, **kwargs)
        return image_filter_adapted
    return decorator