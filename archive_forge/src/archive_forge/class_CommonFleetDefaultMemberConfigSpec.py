from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommonFleetDefaultMemberConfigSpec(_messages.Message):
    """CommonFleetDefaultMemberConfigSpec contains default configuration
  information for memberships of a fleet

  Fields:
    configmanagement: Config Management-specific spec.
    helloworld: Hello World-specific spec.
    identityservice: Identity Service-specific spec.
    mesh: Anthos Service Mesh-specific spec
    policycontroller: Policy Controller spec.
  """
    configmanagement = _messages.MessageField('ConfigManagementMembershipSpec', 1)
    helloworld = _messages.MessageField('HelloWorldMembershipSpec', 2)
    identityservice = _messages.MessageField('IdentityServiceMembershipSpec', 3)
    mesh = _messages.MessageField('ServiceMeshMembershipSpec', 4)
    policycontroller = _messages.MessageField('PolicyControllerMembershipSpec', 5)