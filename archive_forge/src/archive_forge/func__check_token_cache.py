from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def _check_token_cache(self, token_key, duration=None, window_seconds=60):
    token = _session_token_cache.get(token_key, None)
    if token:
        now = datetime.datetime.utcnow()
        expires = boto.utils.parse_ts(token.expiration)
        delta = expires - now
        if delta < datetime.timedelta(seconds=window_seconds):
            msg = 'Cached session token %s is expired' % token_key
            boto.log.debug(msg)
            token = None
    return token