import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _load_auth_context_from_cache(self):
    context = super()._load_auth_context_from_cache()
    if context is None:
        return None
    try:
        self._fetch_auth_token()
    except InvalidCredsError:
        return None
    return context