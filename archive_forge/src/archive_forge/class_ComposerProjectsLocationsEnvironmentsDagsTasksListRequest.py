from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDagsTasksListRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDagsTasksListRequest object.

  Fields:
    pageSize: The maximum number of tasks to return.
    pageToken: The next_page_token returned from a previous List request.
    parent: Required. List tasks in the given parent DAG. Parent must be in
      the form: "projects/{projectId}/locations/{locationId}/environments/{env
      ironmentId}/dags/{dagId}".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)