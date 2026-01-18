import moviepy.video.compositing.transitions as transfx
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    Makes the clip fade in progressively at its own end, this way
    it can be looped indefinitely. ``cross`` is the duration in seconds
    of the fade-in.  