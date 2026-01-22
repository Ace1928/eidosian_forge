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
def _refresh_access_token(self, token):
    keys = ('refreshToken', 'clientId', 'clientSecret', 'registrationExpiresAt')
    missing_keys = [k for k in keys if k not in token]
    if missing_keys:
        msg = f'Unable to refresh SSO token: missing keys: {missing_keys}'
        logger.info(msg)
        return None
    expiry = dateutil.parser.parse(token['registrationExpiresAt'])
    if total_seconds(expiry - self._now()) <= 0:
        logger.info(f'SSO token registration expired at {expiry}')
        return None
    try:
        return self._attempt_create_token(token)
    except ClientError:
        logger.warning('SSO token refresh attempt failed', exc_info=True)
        return None