from moviepy.decorators import apply_to_audio, apply_to_mask

    Returns a clip playing the current clip but at a speed multiplied
    by ``factor``. Instead of factor one can indicate the desired
    ``final_duration`` of the clip, and the factor will be automatically
    computed.
    The same effect is applied to the clip's audio and mask if any.
    