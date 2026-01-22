from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
def _GceMetadataRequest(relative_url, use_metadata_ip=False):
    """Request the given url from the GCE metadata service."""
    if use_metadata_ip:
        base_url = os.environ.get('GCE_METADATA_IP', '169.254.169.254')
    else:
        base_url = os.environ.get('GCE_METADATA_ROOT', 'metadata.google.internal')
    url = 'http://' + base_url + '/computeMetadata/v1/' + relative_url
    headers = {'Metadata-Flavor': 'Google'}
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(request)
    except urllib.error.URLError as e:
        raise exceptions.CommunicationError('Could not reach metadata service: %s' % e.reason)
    return response