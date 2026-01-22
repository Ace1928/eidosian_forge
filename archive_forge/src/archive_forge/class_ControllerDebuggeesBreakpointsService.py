from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddebugger.v2 import clouddebugger_v2_messages as messages
class ControllerDebuggeesBreakpointsService(base_api.BaseApiService):
    """Service class for the controller_debuggees_breakpoints resource."""
    _NAME = 'controller_debuggees_breakpoints'

    def __init__(self, client):
        super(ClouddebuggerV2.ControllerDebuggeesBreakpointsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns the list of all active breakpoints for the debuggee. The breakpoint specification (`location`, `condition`, and `expressions` fields) is semantically immutable, although the field values may change. For example, an agent may update the location line number to reflect the actual line where the breakpoint was set, but this doesn't change the breakpoint semantics. This means that an agent does not need to check if a breakpoint has changed when it encounters the same breakpoint on a successive call. Moreover, an agent should remember the breakpoints that are completed until the controller removes them from the active list to avoid setting those breakpoints again.

      Args:
        request: (ClouddebuggerControllerDebuggeesBreakpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListActiveBreakpointsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='clouddebugger.controller.debuggees.breakpoints.list', ordered_params=['debuggeeId'], path_params=['debuggeeId'], query_params=['agentId', 'successOnTimeout', 'waitToken'], relative_path='v2/controller/debuggees/{debuggeeId}/breakpoints', request_field='', request_type_name='ClouddebuggerControllerDebuggeesBreakpointsListRequest', response_type_name='ListActiveBreakpointsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the breakpoint state or mutable fields. The entire Breakpoint message must be sent back to the controller service. Updates to active breakpoint fields are only allowed if the new value does not change the breakpoint specification. Updates to the `location`, `condition` and `expressions` fields should not alter the breakpoint semantics. These may only make changes such as canonicalizing a value or snapping the location to the correct line of code.

      Args:
        request: (ClouddebuggerControllerDebuggeesBreakpointsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UpdateActiveBreakpointResponse) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='clouddebugger.controller.debuggees.breakpoints.update', ordered_params=['debuggeeId', 'id'], path_params=['debuggeeId', 'id'], query_params=[], relative_path='v2/controller/debuggees/{debuggeeId}/breakpoints/{id}', request_field='updateActiveBreakpointRequest', request_type_name='ClouddebuggerControllerDebuggeesBreakpointsUpdateRequest', response_type_name='UpdateActiveBreakpointResponse', supports_download=False)