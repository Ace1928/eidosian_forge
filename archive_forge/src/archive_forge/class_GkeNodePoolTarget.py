from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeNodePoolTarget(_messages.Message):
    """GKE node pools that Dataproc workloads run on.

  Enums:
    RolesValueListEntryValuesEnum:

  Fields:
    nodePool: Required. The target GKE node pool. Format: 'projects/{project}/
      locations/{location}/clusters/{cluster}/nodePools/{node_pool}'
    nodePoolConfig: Input only. The configuration for the GKE node pool.If
      specified, Dataproc attempts to create a node pool with the specified
      shape. If one with the same name already exists, it is verified against
      all specified fields. If a field differs, the virtual cluster creation
      will fail.If omitted, any node pool with the specified name is used. If
      a node pool with the specified name does not exist, Dataproc create a
      node pool with default values.This is an input only field. It will not
      be returned by the API.
    roles: Required. The roles associated with the GKE node pool.
  """

    class RolesValueListEntryValuesEnum(_messages.Enum):
        """RolesValueListEntryValuesEnum enum type.

    Values:
      ROLE_UNSPECIFIED: Role is unspecified.
      DEFAULT: At least one node pool must have the DEFAULT role. Work
        assigned to a role that is not associated with a node pool is assigned
        to the node pool with the DEFAULT role. For example, work assigned to
        the CONTROLLER role will be assigned to the node pool with the DEFAULT
        role if no node pool has the CONTROLLER role.
      CONTROLLER: Run work associated with the Dataproc control plane (for
        example, controllers and webhooks). Very low resource requirements.
      SPARK_DRIVER: Run work associated with a Spark driver of a job.
      SPARK_EXECUTOR: Run work associated with a Spark executor of a job.
      SHUFFLE_SERVICE: Run work associated with a shuffle service of a job.
        During private preview only, this role must be set explicitly, it does
        not default to DEFAULT. Once the feature reaches public preview, then
        it will default to DEFAULT as the other roles do.
    """
        ROLE_UNSPECIFIED = 0
        DEFAULT = 1
        CONTROLLER = 2
        SPARK_DRIVER = 3
        SPARK_EXECUTOR = 4
        SHUFFLE_SERVICE = 5
    nodePool = _messages.StringField(1)
    nodePoolConfig = _messages.MessageField('GkeNodePoolConfig', 2)
    roles = _messages.EnumField('RolesValueListEntryValuesEnum', 3, repeated=True)