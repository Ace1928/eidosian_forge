from __future__ import annotations
import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping
class OneDayCache(BaseHeuristic):
    """
    Cache the response by providing an expires 1 day in the
    future.
    """

    def update_headers(self, response: HTTPResponse) -> dict[str, str]:
        headers = {}
        if 'expires' not in response.headers:
            date = parsedate(response.headers['date'])
            expires = expire_after(timedelta(days=1), date=datetime(*date[:6], tzinfo=timezone.utc))
            headers['expires'] = datetime_to_header(expires)
            headers['cache-control'] = 'public'
        return headers