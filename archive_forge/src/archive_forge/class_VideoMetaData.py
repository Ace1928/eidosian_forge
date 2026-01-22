import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library
class VideoMetaData:
    __annotations__ = {'has_video': bool, 'video_timebase': Timebase, 'video_duration': float, 'video_fps': float, 'has_audio': bool, 'audio_timebase': Timebase, 'audio_duration': float, 'audio_sample_rate': float}
    __slots__ = ['has_video', 'video_timebase', 'video_duration', 'video_fps', 'has_audio', 'audio_timebase', 'audio_duration', 'audio_sample_rate']

    def __init__(self) -> None:
        self.has_video = False
        self.video_timebase = Timebase(0, 1)
        self.video_duration = 0.0
        self.video_fps = 0.0
        self.has_audio = False
        self.audio_timebase = Timebase(0, 1)
        self.audio_duration = 0.0
        self.audio_sample_rate = 0.0