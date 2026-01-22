import re
import numpy as np
from moviepy.tools import cvsecs
from moviepy.video.VideoClip import TextClip, VideoClip
def add_textclip_if_none(t):
    """ Will generate a textclip if it hasn't been generated asked
            to generate it yet. If there is no subtitle to show at t, return
            false. """
    sub = [((ta, tb), txt) for (ta, tb), txt in self.textclips.keys() if ta <= t < tb]
    if not sub:
        sub = [((ta, tb), txt) for (ta, tb), txt in self.subtitles if ta <= t < tb]
        if not sub:
            return False
    sub = sub[0]
    if sub not in self.textclips.keys():
        self.textclips[sub] = self.make_textclip(sub[1])
    return sub