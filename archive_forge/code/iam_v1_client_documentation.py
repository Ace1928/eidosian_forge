from __future__ import absolute_import
from apitools.base.py import base_api
from samples.iam_sample.iam_v1 import iam_v1_messages as messages
Queries roles that can be granted on a particular resource.

      Args:
        request: (QueryGrantableRolesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryGrantableRolesResponse) The response message.
      