from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsModelsDeleteRequest(_messages.Message):
    """A TranslateProjectsLocationsModelsDeleteRequest object.

  Fields:
    name: Required. The name of the model to delete.
  """
    name = _messages.StringField(1, required=True)