from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyConfig(_messages.Message):
    """The configuration that controls how the data will change.

  Fields:
    imageTransformations: Treat the dataset as an image and redact.
    infoTypeTransformations: Treat the dataset as free-form text and apply the
      same free text transformation everywhere.
    recordTransformations: Treat the dataset as structured. Transformations
      can be applied to specific locations within structured datasets, such as
      transforming a column within a table.
    transformationErrorHandling: Mode for handling transformation errors. If
      left unspecified, the default mode is
      `TransformationErrorHandling.ThrowError`.
  """
    imageTransformations = _messages.MessageField('GooglePrivacyDlpV2ImageTransformations', 1)
    infoTypeTransformations = _messages.MessageField('GooglePrivacyDlpV2InfoTypeTransformations', 2)
    recordTransformations = _messages.MessageField('GooglePrivacyDlpV2RecordTransformations', 3)
    transformationErrorHandling = _messages.MessageField('GooglePrivacyDlpV2TransformationErrorHandling', 4)