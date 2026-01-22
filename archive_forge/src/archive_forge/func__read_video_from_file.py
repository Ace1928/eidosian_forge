import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _read_video_from_file(filename: str, seek_frame_margin: float=0.25, read_video_stream: bool=True, video_width: int=0, video_height: int=0, video_min_dimension: int=0, video_max_dimension: int=0, video_pts_range: Tuple[int, int]=(0, -1), video_timebase: Fraction=default_timebase, read_audio_stream: bool=True, audio_samples: int=0, audio_channels: int=0, audio_pts_range: Tuple[int, int]=(0, -1), audio_timebase: Fraction=default_timebase) -> Tuple[torch.Tensor, torch.Tensor, VideoMetaData]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
    filename (str): path to the video file
    seek_frame_margin (double, optional): seeking frame in the stream is imprecise. Thus,
        when video_start_pts is specified, we seek the pts earlier by seek_frame_margin seconds
    read_video_stream (int, optional): whether read video stream. If yes, set to 1. Otherwise, 0
    video_width/video_height/video_min_dimension/video_max_dimension (int): together decide
        the size of decoded frames:

            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the original frame resolution
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension = 0, keep the aspect ratio and resize the
                frame so that shorter edge size is video_min_dimension
            - When video_width = 0, video_height = 0, video_min_dimension = 0,
                and video_max_dimension != 0, keep the aspect ratio and resize
                the frame so that longer edge size is video_max_dimension
            - When video_width = 0, video_height = 0, video_min_dimension != 0,
                and video_max_dimension != 0, resize the frame so that shorter
                edge size is video_min_dimension, and longer edge size is
                video_max_dimension. The aspect ratio may not be preserved
            - When video_width = 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_height is $video_height
            - When video_width != 0, video_height == 0, video_min_dimension = 0,
                and video_max_dimension = 0, keep the aspect ratio and resize
                the frame so that frame video_width is $video_width
            - When video_width != 0, video_height != 0, video_min_dimension = 0,
                and video_max_dimension = 0, resize the frame so that frame
                video_width and  video_height are set to $video_width and
                $video_height, respectively
    video_pts_range (list(int), optional): the start and end presentation timestamp of video stream
    video_timebase (Fraction, optional): a Fraction rational number which denotes timebase in video stream
    read_audio_stream (int, optional): whether read audio stream. If yes, set to 1. Otherwise, 0
    audio_samples (int, optional): audio sampling rate
    audio_channels (int optional): audio channels
    audio_pts_range (list(int), optional): the start and end presentation timestamp of audio stream
    audio_timebase (Fraction, optional): a Fraction rational number which denotes time base in audio stream

    Returns
        vframes (Tensor[T, H, W, C]): the `T` video frames
        aframes (Tensor[L, K]): the audio frames, where `L` is the number of points and
            `K` is the number of audio_channels
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float)
            and audio_fps (int)
    """
    _validate_pts(video_pts_range)
    _validate_pts(audio_pts_range)
    result = torch.ops.video_reader.read_video_from_file(filename, seek_frame_margin, 0, read_video_stream, video_width, video_height, video_min_dimension, video_max_dimension, video_pts_range[0], video_pts_range[1], video_timebase.numerator, video_timebase.denominator, read_audio_stream, audio_samples, audio_channels, audio_pts_range[0], audio_pts_range[1], audio_timebase.numerator, audio_timebase.denominator)
    vframes, _vframe_pts, vtimebase, vfps, vduration, aframes, aframe_pts, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    if aframes.numel() > 0:
        aframes = _align_audio_frames(aframes, aframe_pts, audio_pts_range)
    return (vframes, aframes, info)