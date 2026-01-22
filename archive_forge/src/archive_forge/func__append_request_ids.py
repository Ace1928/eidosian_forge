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
def _append_request_ids(self, resp):
    """Add request_ids as an attribute to the object

        :param resp: Response object or list of Response objects
        """
    if isinstance(resp, list):
        for resp_obj in resp:
            self._append_request_id(resp_obj)
    elif resp is not None:
        self._append_request_id(resp)