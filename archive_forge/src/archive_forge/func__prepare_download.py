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
def _prepare_download(resp: Response, link: Link, progress_bar: str) -> Iterable[bytes]:
    total_length = _get_http_response_size(resp)
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment
    logged_url = redact_auth_from_url(url)
    if total_length:
        logged_url = f'{logged_url} ({format_size(total_length)})'
    if is_from_cache(resp):
        logger.info('Using cached %s', logged_url)
    else:
        logger.info('Downloading %s', logged_url)
    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > 40 * 1000:
        show_progress = True
    else:
        show_progress = False
    chunks = response_chunks(resp, CONTENT_CHUNK_SIZE)
    if not show_progress:
        return chunks
    renderer = get_download_progress_renderer(bar_type=progress_bar, size=total_length)
    return renderer(chunks)