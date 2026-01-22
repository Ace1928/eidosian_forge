from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@contextlib.contextmanager
def OverrideApiEndpointOverrides(temp_endpoint):
    """Context manager to override securesourcemanager endpoint overrides temporarily.

  Args:
    temp_endpoint: new endpoint value

  Yields:
    None
  """
    endpoint_property = getattr(properties.VALUES.api_endpoint_overrides, 'securesourcemanager')
    old_endpoint = endpoint_property.Get()
    try:
        endpoint_property.Set(temp_endpoint)
        yield
    finally:
        endpoint_property.Set(old_endpoint)