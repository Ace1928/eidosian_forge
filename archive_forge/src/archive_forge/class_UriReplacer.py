from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import peek_iterable
class UriReplacer(peek_iterable.Tap):
    """A Tapper class that replaces each resource item with its URI.

  Attributes:
    _transform_uri: The uri() transform function.
  """

    def __init__(self, transform_uri):
        self._transform_uri = transform_uri

    def Tap(self, resource):
        """Replaces resource with its URI or skips the resource if it has no URI.

    Args:
      resource: The resource to replace with its URI.

    Returns:
      TapInjector(URI, replace=True) if the resource has a URI or False to skip
      the resource.
    """
        if resource_printer_base.IsResourceMarker(resource):
            return True
        uri = self._transform_uri(resource, undefined=None)
        if not uri:
            return False
        return peek_iterable.TapInjector(uri, replace=True)