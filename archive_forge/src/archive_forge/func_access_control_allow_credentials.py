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
@access_control_allow_credentials.setter
def access_control_allow_credentials(self, value: bool | None) -> None:
    if value is True:
        self.headers['Access-Control-Allow-Credentials'] = 'true'
    else:
        self.headers.pop('Access-Control-Allow-Credentials', None)