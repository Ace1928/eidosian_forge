from __future__ import annotations
import typing as t
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from http import HTTPStatus
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..http import dump_cookie
from ..http import HTTP_STATUS_CODES
from ..utils import get_content_type
from werkzeug.datastructures import CallbackDict
from werkzeug.datastructures import ContentRange
from werkzeug.datastructures import ContentSecurityPolicy
from werkzeug.datastructures import ResponseCacheControl
from werkzeug.datastructures import WWWAuthenticate
from werkzeug.http import COEP
from werkzeug.http import COOP
from werkzeug.http import dump_age
from werkzeug.http import dump_header
from werkzeug.http import dump_options_header
from werkzeug.http import http_date
from werkzeug.http import parse_age
from werkzeug.http import parse_cache_control_header
from werkzeug.http import parse_content_range_header
from werkzeug.http import parse_csp_header
from werkzeug.http import parse_date
from werkzeug.http import parse_options_header
from werkzeug.http import parse_set_header
from werkzeug.http import quote_etag
from werkzeug.http import unquote_etag
from werkzeug.utils import header_property
def _clean_status(self, value: str | int | HTTPStatus) -> tuple[str, int]:
    if isinstance(value, (int, HTTPStatus)):
        status_code = int(value)
    else:
        value = value.strip()
        if not value:
            raise ValueError('Empty status argument')
        code_str, sep, _ = value.partition(' ')
        try:
            status_code = int(code_str)
        except ValueError:
            return (f'0 {value}', 0)
        if sep:
            return (value, status_code)
    try:
        status = f'{status_code} {HTTP_STATUS_CODES[status_code].upper()}'
    except KeyError:
        status = f'{status_code} UNKNOWN'
    return (status, status_code)