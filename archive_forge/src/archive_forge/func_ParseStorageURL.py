from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
def ParseStorageURL(self, url, collection=None):
    """Parse gs://bucket/object_path into storage.v1 api resource."""
    match = _GCS_URL_RE.match(url)
    if not match:
        raise InvalidResourceException(url, 'Not a storage url')
    if match.group(2):
        if collection and collection != 'storage.objects':
            raise WrongResourceCollectionException('storage.objects', collection, url)
        return self.ParseResourceId(collection='storage.objects', resource_id=None, kwargs={'bucket': match.group(1), 'object': match.group(2)})
    if collection and collection != 'storage.buckets':
        raise WrongResourceCollectionException('storage.buckets', collection, url)
    return self.ParseResourceId(collection='storage.buckets', resource_id=None, kwargs={'bucket': match.group(1)})