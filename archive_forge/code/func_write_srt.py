import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def write_srt(self, filename):
    with open(filename, 'w+') as f:
        f.write(str(self))