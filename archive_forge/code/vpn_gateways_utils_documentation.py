from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
import six
Sends a SetLabels request for a VPN Gateway and returns the operation.

    Args:
      ref: The VPN Gateway reference object.
      existing_label_fingerprint: The existing label fingerprint.
      new_labels: List of new label key, value pairs.

    Returns:
      The operation reference object for the SetLabels request.
    