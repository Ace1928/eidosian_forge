from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsCatalogTemplatesDeleteRequest(_messages.Message):
    """A DataflowProjectsCatalogTemplatesDeleteRequest object.

  Fields:
    name: name includes project_id and display_name. Delete by
      project_id(pid1) and display_name(tid1). Format:
      projects/{pid1}/catalogTemplates/{tid1}
  """
    name = _messages.StringField(1, required=True)