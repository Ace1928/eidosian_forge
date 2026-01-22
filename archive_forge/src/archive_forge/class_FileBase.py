from __future__ import annotations
import asyncio
import base64
import struct
from io import BytesIO
from pathlib import PurePath
from typing import (
import param
from ..models import PDF as _BkPDF
from ..util import isfile, isurl
from .markup import HTMLBasePane, escape
class FileBase(HTMLBasePane):
    embed = param.Boolean(default=False, doc='\n        Whether to embed the file as base64.')
    filetype: ClassVar[str]
    _extensions: ClassVar[None | Tuple[str, ...]] = None
    _rename: ClassVar[Mapping[str, str | None]] = {'embed': None}
    _rerender_params: ClassVar[List[str]] = ['embed', 'object', 'styles', 'width', 'height']
    __abstract = True

    def __init__(self, object=None, **params):
        if isinstance(object, PurePath):
            object = str(object)
        super().__init__(object=object, **params)

    def _type_error(self, object):
        if isinstance(object, str):
            raise ValueError(f'{type(self).__name__} pane cannot parse string that is not a filename or URL ({object!r}).')
        super()._type_error(object)

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        filetype = cls.filetype.split('+')[0]
        exts = cls._extensions or (filetype,)
        if hasattr(obj, f'_repr_{filetype}_'):
            return 0.15
        if isinstance(obj, PurePath):
            obj = str(obj.absolute())
        if isinstance(obj, str):
            if isurl(obj, exts):
                return True
            elif any((obj.lower().endswith(f'.{ext}') for ext in exts)):
                return True
            elif isurl(obj, None):
                return 0.0
        elif isinstance(obj, bytes):
            try:
                cls._imgshape(obj)
                return True
            except Exception:
                return False
        if hasattr(obj, 'read'):
            return True
        return False

    def _b64(self, data: str | bytes) -> str:
        if not isinstance(data, bytes):
            data = data.encode('utf-8')
        b64 = base64.b64encode(data).decode('utf-8')
        return f'data:image/{self.filetype};base64,{b64}'

    def _data(self, obj: Any) -> bytes | None:
        filetype = self.filetype.split('+')[0]
        if hasattr(obj, f'_repr_{filetype}_'):
            return getattr(obj, f'_repr_{filetype}_')()
        elif isinstance(obj, (str, PurePath)):
            if isfile(obj):
                with open(obj, 'rb') as f:
                    return f.read()
        elif isinstance(obj, bytes):
            return obj
        elif hasattr(obj, 'read'):
            if hasattr(obj, 'seek'):
                obj.seek(0)
            return obj.read()
        elif not isurl(obj, None):
            return None
        from ..io.state import state
        if state._is_pyodide:
            from ..io.pyodide import _IN_WORKER, fetch_binary
            if _IN_WORKER:
                return fetch_binary(obj).read()
            else:
                from pyodide.http import pyfetch

                async def replace_content():
                    self.object = await (await pyfetch(obj)).bytes()
                task = asyncio.create_task(replace_content())
                _tasks.add(task)
                task.add_done_callback(_tasks.discard)
        else:
            import requests
            r = requests.request(url=obj, method='GET')
            return r.content