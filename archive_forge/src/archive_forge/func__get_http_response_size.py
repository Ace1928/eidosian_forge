import email.message
import logging
import mimetypes
import os
from typing import Iterable, Optional, Tuple
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext
def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers['content-length'])
    except (ValueError, KeyError, TypeError):
        return None