from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from six.moves import urllib
@contextlib.contextmanager
def _OverrideEndpoint(override):
    """Context manager to override an API's endpoint overrides for a while."""
    endpoint_property = getattr(properties.VALUES.api_endpoint_overrides, _API_NAME)
    old_endpoint = endpoint_property.Get()
    try:
        endpoint_property.Set(override)
        yield
    finally:
        endpoint_property.Set(old_endpoint)