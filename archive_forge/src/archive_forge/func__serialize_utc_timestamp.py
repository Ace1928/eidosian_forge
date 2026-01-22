import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import NamedTuple, Optional
import dateutil.parser
from dateutil.tz import tzutc
from botocore import UNSIGNED
from botocore.compat import total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.utils import CachedProperty, JSONFileCache, SSOTokenLoader
def _serialize_utc_timestamp(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%dT%H:%M:%SZ')
    return obj