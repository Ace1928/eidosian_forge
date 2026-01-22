from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1NetworkConfig(_messages.Message):
    """Network configuration for a PrivatePool.

  Enums:
    EgressOptionValueValuesEnum: Immutable. Define whether workloads on the
      PrivatePool can talk to public IPs. If unset, the value NO_PUBLIC_EGRESS
      will be used.

  Fields:
    egressOption: Immutable. Define whether workloads on the PrivatePool can
      talk to public IPs. If unset, the value NO_PUBLIC_EGRESS will be used.
    peeredNetwork: Required. Immutable. The network definition that the
      workers are peered to. If this section is left empty, the workers will
      be peered to `WorkerPool.project_id` on the service producer network.
      Must be in the format `projects/{project}/global/networks/{network}`,
      where `{project}` is a project number, such as `12345`, and `{network}`
      is the name of a VPC network in the project. See [Understanding network
      configuration options](https://cloud.google.com/build/docs/private-
      pools/set-up-private-pool-environment)
  """

    class EgressOptionValueValuesEnum(_messages.Enum):
        """Immutable. Define whether workloads on the PrivatePool can talk to
    public IPs. If unset, the value NO_PUBLIC_EGRESS will be used.

    Values:
      EGRESS_OPTION_UNSPECIFIED: Unspecified policy - this is treated as
        NO_PUBLIC_EGRESS.
      NO_PUBLIC_EGRESS: Public egress is disallowed.
      PUBLIC_EGRESS: Public egress is allowed.
    """
        EGRESS_OPTION_UNSPECIFIED = 0
        NO_PUBLIC_EGRESS = 1
        PUBLIC_EGRESS = 2
    egressOption = _messages.EnumField('EgressOptionValueValuesEnum', 1)
    peeredNetwork = _messages.StringField(2)