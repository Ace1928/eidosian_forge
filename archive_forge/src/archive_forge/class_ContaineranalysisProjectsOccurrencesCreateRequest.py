from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOccurrencesCreateRequest(_messages.Message):
    """A ContaineranalysisProjectsOccurrencesCreateRequest object.

  Fields:
    occurrence: A Occurrence resource to be passed as the request body.
    parent: Required. The name of the project in the form of
      `projects/[PROJECT_ID]`, under which the occurrence is to be created.
  """
    occurrence = _messages.MessageField('Occurrence', 1)
    parent = _messages.StringField(2, required=True)