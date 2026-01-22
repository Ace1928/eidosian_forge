from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipFeatureSpec(_messages.Message):
    """MembershipFeatureSpec contains configuration information for a single
  Membership. NOTE: Please use snake case in your feature name.

  Fields:
    anthosobservability: Anthos Observability-specific spec
    cloudbuild: Cloud Build-specific spec
    configmanagement: Config Management-specific spec.
    fleetobservability: Fleet observability membership spec
    helloworld: Hello World-specific spec.
    identityservice: Identity Service-specific spec.
    mesh: Anthos Service Mesh-specific spec
    origin: Whether this per-Membership spec was inherited from a fleet-level
      default. This field can be updated by users by either overriding a
      Membership config (updated to USER implicitly) or setting to FLEET
      explicitly.
    policycontroller: Policy Controller spec.
  """
    anthosobservability = _messages.MessageField('AnthosObservabilityMembershipSpec', 1)
    cloudbuild = _messages.MessageField('MembershipSpec', 2)
    configmanagement = _messages.MessageField('ConfigManagementMembershipSpec', 3)
    fleetobservability = _messages.MessageField('FleetObservabilityMembershipSpec', 4)
    helloworld = _messages.MessageField('HelloWorldMembershipSpec', 5)
    identityservice = _messages.MessageField('IdentityServiceMembershipSpec', 6)
    mesh = _messages.MessageField('ServiceMeshMembershipSpec', 7)
    origin = _messages.MessageField('Origin', 8)
    policycontroller = _messages.MessageField('PolicyControllerMembershipSpec', 9)