from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsPatchRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsPatchRequest object.

  Fields:
    connection: A Connection resource to be passed as the request body.
    name: Output only. Resource name of the Connection. Format:
      projects/{project}/locations/{location}/connections/{connection}
    updateMask: Required. You can modify only the fields listed below. To
      lock/unlock a connection: * `lock_config` To suspend/resume a
      connection: * `suspended` To update the connection details: *
      `description` * `labels` * `connector_version` * `config_variables` *
      `auth_config` * `destination_configs` * `node_config` * `log_config` *
      `ssl_config` * `eventing_enablement_type` * `eventing_config`
  """
    connection = _messages.MessageField('Connection', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)