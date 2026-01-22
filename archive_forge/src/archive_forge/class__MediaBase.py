from __future__ import annotations
import os
from base64 import b64encode
from io import BytesIO
from typing import (
import numpy as np
import param
from ..models import Audio as _BkAudio, Video as _BkVideo
from ..util import isfile, isurl
from .base import ModelPane
class _MediaBase(ModelPane):
    loop = param.Boolean(default=False, doc='\n        Whether the media should loop')
    time = param.Number(default=0, doc='\n        The current timestamp')
    throttle = param.Integer(default=250, doc='\n        How frequently to sample the current playback time in milliseconds')
    paused = param.Boolean(default=True, doc='\n        Whether the media is currently paused')
    object = param.String(default='', allow_None=True, doc='\n        The media file either local or remote.')
    volume = param.Number(default=None, bounds=(0, 100), doc='\n        The volume of the media player.')
    autoplay = param.Boolean(default=False, doc='\n        When True, it specifies that the output will play automatically.\n        In Chromium browsers this requires the user to click play once.')
    muted = param.Boolean(default=False, doc='\n        When True, it specifies that the output should be muted.')
    _default_mime: ClassVar[str]
    _formats: ClassVar[List[str]]
    _media_type: ClassVar[str]
    _rename: ClassVar[Mapping[str, str | None]] = {'sample_rate': None, 'object': 'value'}
    _rerender_params: ClassVar[List[str]] = []
    _updates: ClassVar[bool] = True
    __abstract = True

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        if isinstance(obj, str):
            if isfile(obj) and any((obj.endswith('.' + fmt) for fmt in cls._formats)):
                return True
            if isurl(obj, cls._formats):
                return True
        if hasattr(obj, 'read'):
            return True
        return False

    def _to_np_int16(self, data: np.ndarray):
        dtype = data.dtype
        if dtype in (np.float32, np.float64):
            data = (data * 32768.0).astype(np.int16)
        return data

    def _to_buffer(self, data: np.ndarray | TensorLike):
        if isinstance(data, TensorLike):
            data = data.numpy()
        data = self._to_np_int16(data)
        from scipy.io import wavfile
        buffer = BytesIO()
        wavfile.write(buffer, self.sample_rate, data)
        return buffer

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if 'js_property_callbacks' in msg:
            del msg['js_property_callbacks']
        return msg

    def _transform_object(self, obj: Any) -> Dict[str, Any]:
        fmt = self._default_mime
        if obj is None:
            data = b''
        elif isinstance(obj, (np.ndarray, TensorLike)):
            fmt = 'wav'
            buffer = self._to_buffer(obj)
            data = b64encode(buffer.getvalue())
        elif os.path.isfile(obj):
            fmt = obj.split('.')[-1]
            with open(obj, 'rb') as f:
                data = f.read()
            data = b64encode(data)
        elif obj.lower().startswith('http'):
            return dict(object=obj)
        elif not obj or obj == f'data:{self._media_type}/{fmt};base64,':
            data = b''
        else:
            raise ValueError(f'Object should be either path to a {self._media_type} file or numpy array.')
        b64 = f'data:{self._media_type}/{fmt};base64,{data.decode('utf-8')}'
        return dict(object=b64)