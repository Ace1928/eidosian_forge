from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddebugger.v2 import clouddebugger_v2_messages as messages
class ControllerService(base_api.BaseApiService):
    """Service class for the controller resource."""
    _NAME = 'controller'

    def __init__(self, client):
        super(ClouddebuggerV2.ControllerService, self).__init__(client)
        self._upload_configs = {}