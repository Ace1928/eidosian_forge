from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import peek_iterable
class UriCacher(peek_iterable.Tap):
    """A Tapper class that caches URIs based on the cache update op.

  Attributes:
    _transform_uri: The uri() transform function.
    _update_cache_op: The non-None return value from UpdateUriCache().
    _uris: The list of changed URIs, None if it is corrupt.
  """

    def __init__(self, update_cache_op, transform_uri):
        self._transform_uri = transform_uri
        self._update_cache_op = update_cache_op
        self._uris = []

    def Tap(self, resource):
        """Appends the URI for resource to the list of cache changes.

    Sets self._uris to None if a URI could not be retrieved for any resource.

    Args:
      resource: The resource from which the URI is extracted.

    Returns:
      True - all resources are seen downstream.
    """
        if resource_printer_base.IsResourceMarker(resource):
            return True
        if self._uris is not None:
            uri = self._transform_uri(resource, undefined=None)
            if uri:
                self._uris.append(uri)
            else:
                self._uris = None
        return True

    def Done(self):
        if self._uris is not None:
            self._update_cache_op.Update(self._uris)