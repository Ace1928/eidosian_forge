from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
@retry.RetryOnException(max_retrials=3)
def _ReadNoProxyWithCleanFailures(uri, http_errors_to_ignore=(), timeout=properties.VALUES.compute.gce_metadata_read_timeout_sec.GetInt()):
    """Reads data from a URI with no proxy, yielding cloud-sdk exceptions."""
    try:
        return gce_read.ReadNoProxy(uri, timeout)
    except urllib.error.HTTPError as e:
        if e.code in http_errors_to_ignore:
            return None
        if e.code == 403:
            raise MetadataServerException('The request is rejected. Please check if the metadata server is concealed.\nSee https://cloud.google.com/kubernetes-engine/docs/how-to/protecting-cluster-metadata#concealment for more information about metadata server concealment.')
        raise MetadataServerException(e)
    except urllib.error.URLError as e:
        raise CannotConnectToMetadataServerException(e)