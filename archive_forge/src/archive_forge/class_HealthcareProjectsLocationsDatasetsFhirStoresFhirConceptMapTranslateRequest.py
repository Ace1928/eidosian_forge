from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirConceptMapTranslateRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsFhirStoresFhirConceptMapTranslateRequest
  object.

  Fields:
    code: Required. The code to translate.
    conceptMapVersion: The version of the concept map to use. If unset, the
      most current version is used.
    name: Required. The URL for the concept map to use for the translation.
    system: Required. The system for the code to be translated.
  """
    code = _messages.StringField(1)
    conceptMapVersion = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    system = _messages.StringField(4)