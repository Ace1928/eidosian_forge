from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancePartition(_messages.Message):
    """An isolated set of Cloud Spanner resources that databases can define
  placements on.

  Enums:
    StateValueValuesEnum: Output only. The current instance partition state.

  Fields:
    config: Required. The name of the instance partition's configuration.
      Values are of the form `projects//instanceConfigs/`. See also
      InstanceConfig and ListInstanceConfigs.
    createTime: Output only. The time at which the instance partition was
      created.
    displayName: Required. The descriptive name for this instance partition as
      it appears in UIs. Must be unique per project and between 4 and 30
      characters in length.
    etag: Used for optimistic concurrency control as a way to help prevent
      simultaneous updates of a instance partition from overwriting each
      other. It is strongly suggested that systems make use of the etag in the
      read-modify-write cycle to perform instance partition updates in order
      to avoid race conditions: An etag is returned in the response which
      contains instance partitions, and systems are expected to put that etag
      in the request to update instance partitions to ensure that their change
      will be applied to the same version of the instance partition. If no
      etag is provided in the call to update instance partition, then the
      existing instance partition is overwritten blindly.
    name: Required. A unique identifier for the instance partition. Values are
      of the form `projects//instances//instancePartitions/a-z*[a-z0-9]`. The
      final segment of the name must be between 2 and 64 characters in length.
      An instance partition's name cannot be changed after the instance
      partition is created.
    nodeCount: The number of nodes allocated to this instance partition. Users
      can set the node_count field to specify the target number of nodes
      allocated to the instance partition. This may be zero in API responses
      for instance partitions that are not yet in state `READY`.
    processingUnits: The number of processing units allocated to this instance
      partition. Users can set the processing_units field to specify the
      target number of processing units allocated to the instance partition.
      This may be zero in API responses for instance partitions that are not
      yet in state `READY`.
    referencingBackups: Output only. The names of the backups that reference
      this instance partition. Referencing backups should share the parent
      instance. The existence of any referencing backup prevents the instance
      partition from being deleted.
    referencingDatabases: Output only. The names of the databases that
      reference this instance partition. Referencing databases should share
      the parent instance. The existence of any referencing database prevents
      the instance partition from being deleted.
    state: Output only. The current instance partition state.
    updateTime: Output only. The time at which the instance partition was most
      recently updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current instance partition state.

    Values:
      STATE_UNSPECIFIED: Not specified.
      CREATING: The instance partition is still being created. Resources may
        not be available yet, and operations such as creating placements using
        this instance partition may not work.
      READY: The instance partition is fully created and ready to do work such
        as creating placements and using in databases.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
    config = _messages.StringField(1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    name = _messages.StringField(5)
    nodeCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    processingUnits = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    referencingBackups = _messages.StringField(8, repeated=True)
    referencingDatabases = _messages.StringField(9, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)