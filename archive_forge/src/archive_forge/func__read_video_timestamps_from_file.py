import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
def _read_video_timestamps_from_file(filename: str) -> Tuple[List[int], List[int], VideoMetaData]:
    """
    Decode all video- and audio frames in the video. Only pts
    (presentation timestamp) is returned. The actual frame pixel data is not
    copied. Thus, it is much faster than read_video(...)
    """
    result = torch.ops.video_reader.read_video_from_file(filename, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 1, 1, 0, 0, 0, -1, 0, 1)
    _vframes, vframe_pts, vtimebase, vfps, vduration, _aframes, aframe_pts, atimebase, asample_rate, aduration = result
    info = _fill_info(vtimebase, vfps, vduration, atimebase, asample_rate, aduration)
    vframe_pts = vframe_pts.numpy().tolist()
    aframe_pts = aframe_pts.numpy().tolist()
    return (vframe_pts, aframe_pts, info)