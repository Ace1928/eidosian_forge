from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from http import client as http_client_lib
import json
import logging
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Union
import urllib
from absl import flags
import googleapiclient
from googleapiclient import discovery
import httplib2
import bq_flags
import bq_utils
from clients import bigquery_http
from clients import utils as bq_client_utils
from discovery_documents import discovery_document_cache
from discovery_documents import discovery_document_loader
from utils import bq_api_utils
from utils import bq_error
def GetHttp(self):
    """Returns the httplib2 Http to use."""
    proxy_info = httplib2.proxy_info_from_environment
    if flags.FLAGS.proxy_address and flags.FLAGS.proxy_port:
        try:
            port = int(flags.FLAGS.proxy_port)
        except ValueError:
            raise ValueError('Invalid value for proxy_port: {}'.format(flags.FLAGS.proxy_port))
        proxy_info = httplib2.ProxyInfo(proxy_type=3, proxy_host=flags.FLAGS.proxy_address, proxy_port=port, proxy_user=flags.FLAGS.proxy_username or None, proxy_pass=flags.FLAGS.proxy_password or None)
    http = httplib2.Http(proxy_info=proxy_info, ca_certs=flags.FLAGS.ca_certificates_file or None, disable_ssl_certificate_validation=flags.FLAGS.disable_ssl_validation)
    if hasattr(http, 'redirect_codes'):
        http.redirect_codes = set(http.redirect_codes) - {308}
    if flags.FLAGS.mtls:
        _, self._cert_file = tempfile.mkstemp()
        _, self._key_file = tempfile.mkstemp()
        discovery.add_mtls_creds(http, discovery.get_client_options(), self._cert_file, self._key_file)
    return http