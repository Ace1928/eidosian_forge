from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
class RequestBuilder:
    endpoint = 'http://www.google.com/speech-api/v2/recognize'

    def __init__(self, *, key: str, language: str, filter_level: ProfanityFilterLevel) -> None:
        self.key = key
        self.language = language
        self.filter_level = filter_level

    def build(self, audio_data: AudioData) -> Request:
        if not isinstance(audio_data, AudioData):
            raise ValueError('``audio_data`` must be audio data')
        url = self.build_url()
        headers = self.build_headers(audio_data)
        flac_data = self.build_data(audio_data)
        request = Request(url, data=flac_data, headers=headers)
        return request

    def build_url(self) -> str:
        """
        >>> builder = RequestBuilder(key="awesome-key", language="en-US", filter_level=0)
        >>> builder.build_url()
        'http://www.google.com/speech-api/v2/recognize?client=chromium&lang=en-US&key=awesome-key&pFilter=0'
        """
        params = urlencode({'client': 'chromium', 'lang': self.language, 'key': self.key, 'pFilter': self.filter_level})
        return f'{self.endpoint}?{params}'

    def build_headers(self, audio_data: AudioData) -> RequestHeaders:
        """
        >>> builder = RequestBuilder(key="", language="", filter_level=1)
        >>> audio_data = AudioData(b"", 16_000, 1)
        >>> builder.build_headers(audio_data)
        {'Content-Type': 'audio/x-flac; rate=16000'}
        """
        rate = audio_data.sample_rate
        headers = {'Content-Type': f'audio/x-flac; rate={rate}'}
        return headers

    def build_data(self, audio_data: AudioData) -> bytes:
        flac_data = audio_data.get_flac_data(convert_rate=self.to_convert_rate(audio_data.sample_rate), convert_width=2)
        return flac_data

    @staticmethod
    def to_convert_rate(sample_rate: int) -> int:
        """Audio samples must be at least 8 kHz

        >>> RequestBuilder.to_convert_rate(16_000)
        >>> RequestBuilder.to_convert_rate(8_000)
        >>> RequestBuilder.to_convert_rate(7_999)
        8000
        """
        return None if sample_rate >= 8000 else 8000