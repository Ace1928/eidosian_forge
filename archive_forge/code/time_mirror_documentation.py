from moviepy.decorators import apply_to_audio, apply_to_mask, requires_duration

    Returns a clip that plays the current clip backwards.
    The clip must have its ``duration`` attribute set.
    The same effect is applied to the clip's audio and mask if any.
    