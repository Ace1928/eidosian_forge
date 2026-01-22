import json
from urllib.parse import urlparse, urlunparse
from tornado.log import access_log
from .auth import User
from .prometheus.log_functions import prometheus_log_method
def _scrub_uri(uri: str) -> str:
    """scrub auth info from uri"""
    parsed = urlparse(uri)
    if parsed.query:
        parts = parsed.query.split('&')
        changed = False
        for i, s in enumerate(parts):
            key, sep, value = s.partition('=')
            for substring in _SCRUB_PARAM_KEYS:
                if substring in key:
                    parts[i] = f'{key}{sep}[secret]'
                    changed = True
        if changed:
            parsed = parsed._replace(query='&'.join(parts))
            return urlunparse(parsed)
    return uri