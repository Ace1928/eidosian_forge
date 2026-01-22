from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenomicsProjectsOperationsListRequest(_messages.Message):
    """A GenomicsProjectsOperationsListRequest object.

  Fields:
    filter: A string for filtering Operations. In v2alpha1, the following
      filter fields are supported: * createTime: The time this job was created
      * events: The set of event (names) that have occurred while running the
      pipeline. The : operator can be used to determine if a particular event
      has occurred. * error: If the pipeline is running, this value is NULL.
      Once the pipeline finishes, the value is the standard Google error code.
      * labels.key or labels."key with space" where key is a label key. *
      done: If the pipeline is running, this value is false. Once the pipeline
      finishes, the value is true. Examples: * `projectId = my-project AND
      createTime >= 1432140000` * `projectId = my-project AND createTime >=
      1432140000 AND createTime <= 1432150000 AND status = RUNNING` *
      `projectId = my-project AND labels.color = *` * `projectId = my-project
      AND labels.color = red`
    name: The name of the operation's parent resource.
    pageSize: The maximum number of results to return. The maximum value is
      256.
    pageToken: The standard list page token.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)