from moviepy.decorators import apply_to_audio, apply_to_mask, requires_duration
from moviepy.video.compositing.concatenate import concatenate_videoclips
from .time_mirror import time_mirror

    Returns a clip that plays the current clip once forwards and
    then once backwards. This is very practival to make video that
    loop well, e.g. to create animated GIFs.
    This effect is automatically applied to the clip's mask and audio
    if they exist.
    