from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddebugger.v2 import clouddebugger_v2_messages as messages
Lists all the debuggees that the user has access to.

      Args:
        request: (ClouddebuggerDebuggerDebuggeesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDebuggeesResponse) The response message.
      