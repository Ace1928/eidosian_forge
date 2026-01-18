import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
@requires_duration
def write_audiofile(self, filename, fps=None, nbytes=2, buffersize=2000, codec=None, bitrate=None, ffmpeg_params=None, write_logfile=False, verbose=True, logger='bar'):
    """ Writes an audio file from the AudioClip.


        Parameters
        -----------

        filename
          Name of the output file

        fps
          Frames per second. If not set, it will try default to self.fps if
          already set, otherwise it will default to 44100

        nbytes
          Sample width (set to 2 for 16-bit sound, 4 for 32-bit sound)

        codec
          Which audio codec should be used. If None provided, the codec is
          determined based on the extension of the filename. Choose
          'pcm_s16le' for 16-bit wav and 'pcm_s32le' for 32-bit wav.

        bitrate
          Audio bitrate, given as a string like '50k', '500k', '3000k'.
          Will determine the size and quality of the output file.
          Note that it mainly an indicative goal, the bitrate won't
          necessarily be the this in the output file.

        ffmpeg_params
          Any additional parameters you would like to pass, as a list
          of terms, like ['-option1', 'value1', '-option2', 'value2']

        write_logfile
          If true, produces a detailed logfile named filename + '.log'
          when writing the file

        verbose
          Boolean indicating whether to print infomation
          
        logger
          Either 'bar' or None or any Proglog logger

        """
    if not fps:
        if not self.fps:
            fps = 44100
        else:
            fps = self.fps
    if codec is None:
        name, ext = os.path.splitext(os.path.basename(filename))
        try:
            codec = extensions_dict[ext[1:]]['codec'][0]
        except KeyError:
            raise ValueError("MoviePy couldn't find the codec associated with the filename. Provide the 'codec' parameter in write_audiofile.")
    return ffmpeg_audiowrite(self, filename, fps, nbytes, buffersize, codec=codec, bitrate=bitrate, write_logfile=write_logfile, verbose=verbose, ffmpeg_params=ffmpeg_params, logger=logger)