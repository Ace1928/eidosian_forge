import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
def _check_placement_api_available(f):
    """Check if the placement API is available.

    :param f: Function to execute.
    :returns: The returned value of the function f.
    """

    @functools.wraps(f)
    def wrapper(self, *a, **k):
        try:
            if not self._client:
                self._client = self._create_client()
            return f(self, *a, **k)
        except ks_exc.http.HttpError as exc:
            if 400 <= exc.http_status <= 499:
                raise n_exc.PlacementClientError(msg=exc.response.text.replace('\n', ' '))
            raise
    return wrapper