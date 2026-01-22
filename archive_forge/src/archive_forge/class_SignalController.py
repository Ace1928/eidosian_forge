from heat.api.aws import exception
from heat.common import identifier
from heat.common import wsgi
from heat.rpc import client as rpc_client
class SignalController(object):

    def __init__(self, options):
        self.options = options
        self.rpc_client = rpc_client.EngineClient()

    def update_waitcondition(self, req, body, arn):
        con = req.context
        identity = identifier.ResourceIdentifier.from_arn(arn)
        try:
            md = self.rpc_client.resource_signal(con, stack_identity=dict(identity.stack()), resource_name=identity.resource_name, details=body, sync_call=True)
        except Exception as ex:
            return exception.map_remote_error(ex)
        return {'resource': identity.resource_name, 'metadata': md}

    def signal(self, req, arn, body=None):
        con = req.context
        identity = identifier.ResourceIdentifier.from_arn(arn)
        try:
            self.rpc_client.resource_signal(con, stack_identity=dict(identity.stack()), resource_name=identity.resource_name, details=body)
        except Exception as ex:
            return exception.map_remote_error(ex)