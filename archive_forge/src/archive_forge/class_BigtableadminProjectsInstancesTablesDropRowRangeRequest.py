from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesDropRowRangeRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesDropRowRangeRequest object.

  Fields:
    dropRowRangeRequest: A DropRowRangeRequest resource to be passed as the
      request body.
    name: Required. The unique name of the table on which to drop a range of
      rows. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}`.
  """
    dropRowRangeRequest = _messages.MessageField('DropRowRangeRequest', 1)
    name = _messages.StringField(2, required=True)