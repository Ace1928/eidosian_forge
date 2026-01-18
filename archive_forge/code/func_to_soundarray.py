import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
@requires_duration
def to_soundarray(self, tt=None, fps=None, quantize=False, nbytes=2, buffersize=50000):
    """
        Transforms the sound into an array that can be played by pygame
        or written in a wav file. See ``AudioClip.preview``.
        
        Parameters
        ------------
        
        fps
          Frame rate of the sound for the conversion.
          44100 for top quality.
        
        nbytes
          Number of bytes to encode the sound: 1 for 8bit sound,
          2 for 16bit, 4 for 32bit sound.
          
        """
    if fps is None:
        fps = self.fps
    stacker = np.vstack if self.nchannels == 2 else np.hstack
    max_duration = 1.0 * buffersize / fps
    if tt is None:
        if self.duration > max_duration:
            return stacker(self.iter_chunks(fps=fps, quantize=quantize, nbytes=2, chunksize=buffersize))
        else:
            tt = np.arange(0, self.duration, 1.0 / fps)
    '\n        elif len(tt)> 1.5*buffersize:\n            nchunks = int(len(tt)/buffersize+1)\n            tt_chunks = np.array_split(tt, nchunks)\n            return stacker([self.to_soundarray(tt=ttc, buffersize=buffersize, fps=fps,\n                                        quantize=quantize, nbytes=nbytes)\n                              for ttc in tt_chunks])\n        '
    snd_array = self.get_frame(tt)
    if quantize:
        snd_array = np.maximum(-0.99, np.minimum(0.99, snd_array))
        inttype = {1: 'int8', 2: 'int16', 4: 'int32'}[nbytes]
        snd_array = (2 ** (8 * nbytes - 1) * snd_array).astype(inttype)
    return snd_array