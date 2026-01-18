import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def to_srt(sub_element):
    (ta, tb), txt = sub_element
    fta = cvsecs(ta)
    ftb = cvsecs(tb)
    return '%s - %s\n%s' % (fta, ftb, txt)