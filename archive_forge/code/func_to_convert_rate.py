from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
@staticmethod
def to_convert_rate(sample_rate: int) -> int:
    """Audio samples must be at least 8 kHz

        >>> RequestBuilder.to_convert_rate(16_000)
        >>> RequestBuilder.to_convert_rate(8_000)
        >>> RequestBuilder.to_convert_rate(7_999)
        8000
        """
    return None if sample_rate >= 8000 else 8000