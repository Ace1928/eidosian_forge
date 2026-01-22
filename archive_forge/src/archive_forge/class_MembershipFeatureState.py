from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipFeatureState(_messages.Message):
    """MembershipFeatureState contains Feature status information for a single
  Membership.

  Fields:
    appdevexperience: Appdevexperience specific state.
    clusterupgrade: ClusterUpgrade state.
    configmanagement: Config Management-specific state.
    fleetobservability: Fleet observability membership state.
    helloworld: Hello World-specific state.
    identityservice: Identity Service-specific state.
    metering: Metering-specific state.
    policycontroller: Policycontroller-specific state.
    servicemesh: Service Mesh-specific state.
    state: The high-level state of this Feature for a single membership.
  """
    appdevexperience = _messages.MessageField('AppDevExperienceFeatureState', 1)
    clusterupgrade = _messages.MessageField('ClusterUpgradeMembershipState', 2)
    configmanagement = _messages.MessageField('ConfigManagementMembershipState', 3)
    fleetobservability = _messages.MessageField('FleetObservabilityMembershipState', 4)
    helloworld = _messages.MessageField('HelloWorldMembershipState', 5)
    identityservice = _messages.MessageField('IdentityServiceMembershipState', 6)
    metering = _messages.MessageField('MeteringMembershipState', 7)
    policycontroller = _messages.MessageField('PolicyControllerMembershipState', 8)
    servicemesh = _messages.MessageField('ServiceMeshMembershipState', 9)
    state = _messages.MessageField('FeatureState', 10)