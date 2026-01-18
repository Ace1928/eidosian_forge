from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
Updates a Monitoring Snooze.

    If fields is not specified, then the snooze is replaced entirely. If
    fields are specified, then only the fields are replaced.

    Args:
      snooze_ref: resources.Resource, Resource reference to the snooze to be
          updated.
      snooze: Snooze, The snooze message to be sent with the request.
      fields: str, Comma separated list of field masks.
    Returns:
      Snooze, The updated Snooze.
    