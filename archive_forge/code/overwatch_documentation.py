from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.scc.slz_overwatch import instances
Implements method for the overwatch command  `operation`.

    Args:
      operation_id: The operation ID of google.lonrunning.operation. Format:
        organizations/<ORG_ID>/locations/<LOCATION_ID>/operations/<OPERATION_ID>.

    Returns:
      response: The json response from the Operation method of API client.
    