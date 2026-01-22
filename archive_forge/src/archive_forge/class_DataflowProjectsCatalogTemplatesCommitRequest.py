from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsCatalogTemplatesCommitRequest(_messages.Message):
    """A DataflowProjectsCatalogTemplatesCommitRequest object.

  Fields:
    commitTemplateVersionRequest: A CommitTemplateVersionRequest resource to
      be passed as the request body.
    name: The location of the template, name includes project_id and
      display_name. Commit using project_id(pid1) and display_name(tid1).
      Format: projects/{pid1}/catalogTemplates/{tid1}
  """
    commitTemplateVersionRequest = _messages.MessageField('CommitTemplateVersionRequest', 1)
    name = _messages.StringField(2, required=True)