from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateTemplateVersionRequest(_messages.Message):
    """Creates a new Template with TemplateVersions.

  Fields:
    templateVersion: The TemplateVersion object to create.
  """
    templateVersion = _messages.MessageField('TemplateVersion', 1)