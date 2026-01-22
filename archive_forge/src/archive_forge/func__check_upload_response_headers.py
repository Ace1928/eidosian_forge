import http.client as http
import os
import sys
import urllib.parse as urlparse
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
from webob import exc
from glance.common import config
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def _check_upload_response_headers(headers, body):
    """Check that the headers of an upload are reasonable.

    headers: the headers from the upload
    body: the body from the upload
    """
    if 'status' not in headers:
        try:
            d = jsonutils.loads(body)
            if 'image' in d and 'status' in d['image']:
                return
        except Exception:
            raise exception.UploadException(body)