from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
Gets the IAM policy for a DeviceRegistry.

    Args:
      registry_ref: a Resource reference to a
        cloudiot.projects.locations.registries resource.

    Returns:
      Policy: the policy for the device registry.
    