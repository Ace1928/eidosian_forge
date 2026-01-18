from __future__ import annotations
import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping
Update the response headers with any new headers.

        NOTE: This SHOULD always include some Warning header to
              signify that the response was cached by the client, not
              by way of the provided headers.
        