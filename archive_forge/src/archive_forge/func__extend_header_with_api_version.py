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
def _extend_header_with_api_version(self, **kwargs):
    headers = kwargs.get('headers', {})
    if API_VERSION_REQUEST_HEADER not in headers:
        if 'headers' not in kwargs:
            kwargs['headers'] = self._api_version_header
        else:
            kwargs['headers'].update(self._api_version_header)
    return kwargs