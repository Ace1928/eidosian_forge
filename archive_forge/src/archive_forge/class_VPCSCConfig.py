from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VPCSCConfig(_messages.Message):
    """The Artifact Registry VPC SC config that apply to a Project.

  Enums:
    VpcscPolicyValueValuesEnum: The project per location VPC SC policy that
      defines the VPC SC behavior for the Remote Repository (Allow/Deny).

  Fields:
    name: The name of the project's VPC SC Config. Always of the form:
      projects/{projectID}/locations/{location}/vpcscConfig In update request:
      never set In response: always set
    vpcscPolicy: The project per location VPC SC policy that defines the VPC
      SC behavior for the Remote Repository (Allow/Deny).
  """

    class VpcscPolicyValueValuesEnum(_messages.Enum):
        """The project per location VPC SC policy that defines the VPC SC
    behavior for the Remote Repository (Allow/Deny).

    Values:
      VPCSC_POLICY_UNSPECIFIED: VPCSC_POLICY_UNSPECIFIED - the VPS SC policy
        is not defined. When VPS SC policy is not defined - the Service will
        use the default behavior (VPCSC_DENY).
      DENY: VPCSC_DENY - repository will block the requests to the Upstreams
        for the Remote Repositories if the resource is in the perimeter.
      ALLOW: VPCSC_ALLOW - repository will allow the requests to the Upstreams
        for the Remote Repositories if the resource is in the perimeter.
    """
        VPCSC_POLICY_UNSPECIFIED = 0
        DENY = 1
        ALLOW = 2
    name = _messages.StringField(1)
    vpcscPolicy = _messages.EnumField('VpcscPolicyValueValuesEnum', 2)