from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsAnnotationStoresDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsAnnotationStoresDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the Annotation store to delete.
  """
    name = _messages.StringField(1, required=True)