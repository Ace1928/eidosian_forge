import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _read_video(filename: str, start_pts: Union[float, Fraction]=0, end_pts: Optional[Union[float, Fraction]]=None, pts_unit: str='pts') -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if end_pts is None:
        end_pts = float('inf')
    if pts_unit == 'pts':
        warnings.warn("The pts_unit 'pts' gives wrong results and will be removed in a " + "follow-up version. Please use pts_unit 'sec'.")
    info = _probe_video_from_file(filename)
    has_video = info.has_video
    has_audio = info.has_audio

    def get_pts(time_base):
        start_offset = start_pts
        end_offset = end_pts
        if pts_unit == 'sec':
            start_offset = int(math.floor(start_pts * (1 / time_base)))
            if end_offset != float('inf'):
                end_offset = int(math.ceil(end_pts * (1 / time_base)))
        if end_offset == float('inf'):
            end_offset = -1
        return (start_offset, end_offset)
    video_pts_range = (0, -1)
    video_timebase = default_timebase
    if has_video:
        video_timebase = Fraction(info.video_timebase.numerator, info.video_timebase.denominator)
        video_pts_range = get_pts(video_timebase)
    audio_pts_range = (0, -1)
    audio_timebase = default_timebase
    if has_audio:
        audio_timebase = Fraction(info.audio_timebase.numerator, info.audio_timebase.denominator)
        audio_pts_range = get_pts(audio_timebase)
    vframes, aframes, info = _read_video_from_file(filename, read_video_stream=True, video_pts_range=video_pts_range, video_timebase=video_timebase, read_audio_stream=True, audio_pts_range=audio_pts_range, audio_timebase=audio_timebase)
    _info = {}
    if has_video:
        _info['video_fps'] = info.video_fps
    if has_audio:
        _info['audio_fps'] = info.audio_sample_rate
    return (vframes, aframes, _info)