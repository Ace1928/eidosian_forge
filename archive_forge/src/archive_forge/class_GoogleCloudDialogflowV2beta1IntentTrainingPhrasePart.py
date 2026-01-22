from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentTrainingPhrasePart(_messages.Message):
    """Represents a part of a training phrase.

  Fields:
    alias: Optional. The parameter name for the value extracted from the
      annotated part of the example. This field is required for annotated
      parts of the training phrase.
    entityType: Optional. The entity type name prefixed with `@`. This field
      is required for annotated parts of the training phrase.
    text: Required. The text for this part.
    userDefined: Optional. Indicates whether the text was manually annotated.
      This field is set to true when the Dialogflow Console is used to
      manually annotate the part. When creating an annotated part with the
      API, you must set this to true.
  """
    alias = _messages.StringField(1)
    entityType = _messages.StringField(2)
    text = _messages.StringField(3)
    userDefined = _messages.BooleanField(4)