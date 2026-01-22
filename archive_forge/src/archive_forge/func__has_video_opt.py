import io
import warnings
from typing import Any, Dict, Iterator, Optional
import torch
from ..utils import _log_api_usage_once
from ._video_opt import _HAS_VIDEO_OPT
def _has_video_opt() -> bool:
    return False