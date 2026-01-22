from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesConnectionsListRequest(_messages.Message):
    """A ServicenetworkingServicesConnectionsListRequest object.

  Fields:
    network: Network name in the consumer project.   This network must have
      been already peered with a shared VPC network using CreateConnection
      method. Must be in a form
      'projects/{project}/global/networks/{network}'. {project} is a project
      number, as in '12345' {network} is network name.
    parent: Provider peering service that is managing peering connectivity for
      a service provider organization. For Google services that support this
      functionality it is 'services/servicenetworking.googleapis.com'.
  """
    network = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)