import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def DetectGce():
    """Determine whether or not we're running on GCE.

    This is based on:
      https://cloud.google.com/compute/docs/metadata#runninggce

    Returns:
      True iff we're running on a GCE instance.
    """
    metadata_url = 'http://{}'.format(os.environ.get('GCE_METADATA_ROOT', 'metadata.google.internal'))
    try:
        o = urllib_request.build_opener(urllib_request.ProxyHandler({})).open(urllib_request.Request(metadata_url, headers={'Metadata-Flavor': 'Google'}))
    except urllib_error.URLError:
        return False
    return o.getcode() == http_client.OK and o.headers.get('metadata-flavor') == 'Google'