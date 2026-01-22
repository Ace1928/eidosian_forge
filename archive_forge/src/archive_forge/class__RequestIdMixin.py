import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
class _RequestIdMixin(object):
    """Wrapper class to expose x-openstack-request-id to the caller."""

    def _request_ids_setup(self):
        self._request_ids = []

    @property
    def request_ids(self):
        return self._request_ids

    def _append_request_ids(self, resp):
        """Add request_ids as an attribute to the object

        :param resp: Response object or list of Response objects
        """
        if isinstance(resp, list):
            for resp_obj in resp:
                self._append_request_id(resp_obj)
        elif resp is not None:
            self._append_request_id(resp)

    def _append_request_id(self, resp):
        if isinstance(resp, requests.Response):
            request_id = resp.headers.get('x-openstack-request-id')
        else:
            request_id = resp
        if request_id:
            self._request_ids.append(request_id)