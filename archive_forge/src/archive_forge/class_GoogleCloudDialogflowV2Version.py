from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Version(_messages.Message):
    """You can create multiple versions of your agent and publish them to
  separate environments. When you edit an agent, you are editing the draft
  agent. At any point, you can save the draft agent as an agent version, which
  is an immutable snapshot of your agent. When you save the draft agent, it is
  published to the default environment. When you create agent versions, you
  can publish them to custom environments. You can create a variety of custom
  environments for: - testing - development - production - etc. For more
  information, see the [versions and environments
  guide](https://cloud.google.com/dialogflow/docs/agents-versions).

  Enums:
    StatusValueValuesEnum: Output only. The status of this version. This field
      is read-only and cannot be set by create and update methods.

  Fields:
    createTime: Output only. The creation time of this version. This field is
      read-only, i.e., it cannot be set by create and update methods.
    description: Optional. The developer-provided description of this version.
    name: Output only. The unique identifier of this agent version. Supported
      formats: - `projects//agent/versions/` -
      `projects//locations//agent/versions/`
    status: Output only. The status of this version. This field is read-only
      and cannot be set by create and update methods.
    versionNumber: Output only. The sequential number of this version. This
      field is read-only which means it cannot be set by create and update
      methods.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Output only. The status of this version. This field is read-only and
    cannot be set by create and update methods.

    Values:
      VERSION_STATUS_UNSPECIFIED: Not specified. This value is not used.
      IN_PROGRESS: Version is not ready to serve (e.g. training is in
        progress).
      READY: Version is ready to serve.
      FAILED: Version training failed.
    """
        VERSION_STATUS_UNSPECIFIED = 0
        IN_PROGRESS = 1
        READY = 2
        FAILED = 3
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    status = _messages.EnumField('StatusValueValuesEnum', 4)
    versionNumber = _messages.IntegerField(5, variant=_messages.Variant.INT32)