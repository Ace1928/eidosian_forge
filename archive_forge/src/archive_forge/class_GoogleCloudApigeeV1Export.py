from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Export(_messages.Message):
    """Details of an export job.

  Fields:
    created: Output only. Time the export job was created.
    datastoreName: Name of the datastore that is the destination of the export
      job [datastore]
    description: Description of the export job.
    error: Output only. Error is set when export fails
    executionTime: Output only. Execution time for this export job. If the job
      is still in progress, it will be set to the amount of time that has
      elapsed since`created`, in seconds. Else, it will set to (`updated` -
      `created`), in seconds.
    name: Display name of the export job.
    self: Output only. Self link of the export job. A URI that can be used to
      retrieve the status of an export job. Example: `/organizations/myorg/env
      ironments/myenv/analytics/exports/9cfc0d85-0f30-46d6-ae6f-318d0cb961bd`
    state: Output only. Status of the export job. Valid values include
      `enqueued`, `running`, `completed`, and `failed`.
    updated: Output only. Time the export job was last updated.
  """
    created = _messages.StringField(1)
    datastoreName = _messages.StringField(2)
    description = _messages.StringField(3)
    error = _messages.StringField(4)
    executionTime = _messages.StringField(5)
    name = _messages.StringField(6)
    self = _messages.StringField(7)
    state = _messages.StringField(8)
    updated = _messages.StringField(9)