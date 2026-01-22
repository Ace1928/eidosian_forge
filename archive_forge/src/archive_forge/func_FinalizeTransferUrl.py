import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def FinalizeTransferUrl(self, url):
    """Modify the url for a given transfer, based on auth and version."""
    url_builder = _UrlBuilder.FromUrl(url)
    if getattr(self.global_params, 'key', None):
        url_builder.query_params['key'] = self.global_params.key
    if self.overwrite_transfer_urls_with_client_base:
        client_url_builder = _UrlBuilder.FromUrl(self._url)
        url_builder.base_url = client_url_builder.base_url
    return url_builder.url