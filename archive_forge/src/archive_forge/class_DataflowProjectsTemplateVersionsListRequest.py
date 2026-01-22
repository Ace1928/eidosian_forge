from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsTemplateVersionsListRequest(_messages.Message):
    """A DataflowProjectsTemplateVersionsListRequest object.

  Fields:
    pageSize: The maximum number of TemplateVersions to return per page.
    pageToken: The page token, received from a previous ListTemplateVersions
      call. Provide this to retrieve the subsequent page.
    parent: parent includes project_id, and display_name is optional. List by
      project_id(pid1) and display_name(tid1). Format:
      projects/{pid1}/catalogTemplates/{tid1} List by project_id(pid1).
      Format: projects/{pid1}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)