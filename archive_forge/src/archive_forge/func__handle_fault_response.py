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
def _handle_fault_response(self, status_code, response_body, resp):
    _logger.debug('Error message: %s', response_body)
    try:
        des_error_body = self.deserialize(response_body, status_code)
    except Exception:
        des_error_body = {'message': response_body}
    error_body = self._convert_into_with_meta(des_error_body, resp)
    exception_handler_v20(status_code, error_body)