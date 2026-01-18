import numpy as np
from .avconv import LibAVReader
from .avconv import LibAVWriter
from .ffmpeg import FFmpegReader
from .ffmpeg import FFmpegWriter
from .. import _HAS_AVCONV
from .. import _HAS_FFMPEG
from ..utils import *
def vread(fname, height=0, width=0, num_frames=0, as_grey=False, inputdict=None, outputdict=None, backend='ffmpeg', verbosity=0):
    """Load a video from file entirely into memory.

    Parameters
    ----------
    fname : string
        Video file name, e.g. ``bickbuckbunny.mp4``

    height : int
        Set the source video height used for decoding.
        Useful for raw inputs when video header does not exist.

    width : int
        Set the source video width used for decoding.
        Useful for raw inputs when video header does not exist.

    num_frames : int
        Only read the first `num_frames` number of frames from video.
        Setting `num_frames` to small numbers can significantly
        speed up video loading times.

    as_grey : bool
        If true, only load the luminance channel of the input video.

    inputdict : dict
        Input dictionary parameters, i.e. how to interpret the input file.

    outputdict : dict
        Output dictionary parameters, i.e. how to encode the data
        when sending back to the python process.

    backend : string
        Program to use for handling video data.
        Only 'ffmpeg' and 'libav' are supported at this time.

    verbosity : int
        Setting to 0 (default) disables all debugging output.
        Setting to 1 enables all debugging output.
        Useful to see if the backend is behaving properly.

    Returns
    -------
    vid_array : ndarray
        ndarray of dimension (T, M, N, C), where T
        is the number of frames, M is the height, N is
        width, and C is depth.

    """
    if not inputdict:
        inputdict = {}
    if not outputdict:
        outputdict = {}
    if backend == 'ffmpeg':
        assert _HAS_FFMPEG, 'Cannot find installation of real FFmpeg (which comes with ffprobe).'
        if height != 0 and width != 0:
            inputdict['-s'] = str(width) + 'x' + str(height)
        if num_frames != 0:
            outputdict['-vframes'] = str(num_frames)
        if as_grey:
            outputdict['-pix_fmt'] = 'gray'
        reader = FFmpegReader(fname, inputdict=inputdict, outputdict=outputdict, verbosity=verbosity)
        T, M, N, C = reader.getShape()
        videodata = np.empty((T, M, N, C), dtype=reader.dtype)
        for idx, frame in enumerate(reader.nextFrame()):
            videodata[idx, :, :, :] = frame
        if as_grey:
            videodata = vshape(videodata[:, :, :, 0])
        reader.close()
        return videodata
    elif backend == 'libav':
        assert _HAS_AVCONV, 'Cannot find installation of libav.'
        if height != 0 and width != 0:
            inputdict['-s'] = str(width) + 'x' + str(height)
        if num_frames != 0:
            outputdict['-vframes'] = str(num_frames)
        reader = LibAVReader(fname, inputdict=inputdict, outputdict=outputdict, verbosity=verbosity)
        T, M, N, C = reader.getShape()
        videodata = np.empty((T, M, N, C), dtype=reader.dtype)
        for idx, frame in enumerate(reader.nextFrame()):
            videodata[idx, :, :, :] = frame
        reader.close()
        return videodata
    else:
        raise NotImplemented