from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddRolesRequest(_messages.Message):
    """Request for AddRoles to allow Service Producers to add roles in the
  shared VPC host project for them to use.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is a project
      number, as in '12345' {network} is a network name.
    policyBinding: Required. List of policy bindings to add to shared VPC host
      project.
  """
    consumerNetwork = _messages.StringField(1)
    policyBinding = _messages.MessageField('PolicyBinding', 2, repeated=True)