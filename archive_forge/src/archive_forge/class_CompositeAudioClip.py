import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
class CompositeAudioClip(AudioClip):
    """ Clip made by composing several AudioClips.
    
    An audio clip made by putting together several audio clips.
    
    Parameters
    ------------
    
    clips
      List of audio clips, which may start playing at different times or
      together. If all have their ``duration`` attribute set, the
      duration of the composite clip is computed automatically.
    
    """

    def __init__(self, clips):
        Clip.__init__(self)
        self.clips = clips
        ends = [c.end for c in self.clips]
        self.nchannels = max([c.nchannels for c in self.clips])
        if not any([e is None for e in ends]):
            self.duration = max(ends)
            self.end = max(ends)

        def make_frame(t):
            played_parts = [c.is_playing(t) for c in self.clips]
            sounds = [c.get_frame(t - c.start) * np.array([part]).T for c, part in zip(self.clips, played_parts) if part is not False]
            if isinstance(t, np.ndarray):
                zero = np.zeros((len(t), self.nchannels))
            else:
                zero = np.zeros(self.nchannels)
            return zero + sum(sounds)
        self.make_frame = make_frame