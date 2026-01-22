import warnings
from typing import Dict, Optional, Union
from .api import from_bytes, from_fp, from_path, normalize
from .constant import CHARDET_CORRESPONDENCE
from .models import CharsetMatch, CharsetMatches
class CharsetDetector(CharsetNormalizerMatches):
    pass