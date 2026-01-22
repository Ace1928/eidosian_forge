import os
import subprocess as sp
import proglog
from moviepy.compat import DEVNULL
from moviepy.config import get_setting
from moviepy.decorators import requires_duration

    A function that wraps the FFMPEG_AudioWriter to write an AudioClip
    to a file.

    NOTE: verbose is deprecated.
    