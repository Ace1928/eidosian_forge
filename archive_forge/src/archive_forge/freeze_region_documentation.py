from moviepy.decorators import apply_to_mask
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from .crop import crop
 Freezes one region of the clip while the rest remains animated.
    
    You can choose one of three methods by providing either `region`,
    `outside_region`, or `mask`.

    Parameters
    -----------

    t
      Time at which to freeze the freezed region.

    region
      A tuple (x1, y1, x2, y2) defining the region of the screen (in pixels)
      which will be freezed. You can provide outside_region or mask instead.

    outside_region
      A tuple (x1, y1, x2, y2) defining the region of the screen (in pixels)
      which will be the only non-freezed region.

    mask
      If not None, will overlay a freezed version of the clip on the current clip,
      with the provided mask. In other words, the "visible" pixels in the mask
      indicate the freezed region in the final picture.

    