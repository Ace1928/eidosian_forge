from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.metadata import BaseDistribution, MemoryWheel, get_wheel_distribution
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
def _stream_response(self, start: int, end: int, base_headers: Dict[str, str]=HEADERS) -> Response:
    """Return HTTP response to a range request from start to end."""
    headers = base_headers.copy()
    headers['Range'] = f'bytes={start}-{end}'
    headers['Cache-Control'] = 'no-cache'
    return self._session.get(self._url, headers=headers, stream=True)