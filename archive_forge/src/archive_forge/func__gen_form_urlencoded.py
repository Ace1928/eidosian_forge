import io
from typing import Any, Iterable, List, Optional
from urllib.parse import urlencode
from multidict import MultiDict, MultiDictProxy
from . import hdrs, multipart, payload
from .helpers import guess_filename
from .payload import Payload
def _gen_form_urlencoded(self) -> payload.BytesPayload:
    data = []
    for type_options, _, value in self._fields:
        data.append((type_options['name'], value))
    charset = self._charset if self._charset is not None else 'utf-8'
    if charset == 'utf-8':
        content_type = 'application/x-www-form-urlencoded'
    else:
        content_type = 'application/x-www-form-urlencoded; charset=%s' % charset
    return payload.BytesPayload(urlencode(data, doseq=True, encoding=charset).encode(), content_type=content_type)