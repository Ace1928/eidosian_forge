from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirConceptMapSearchTranslateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirConceptMapSearchTrans
  lateRequest object.

  Fields:
    code: Required. The code to translate.
    conceptMapVersion: The version of the concept map to use. If unset, the
      most current version is used.
    parent: Required. The name for the FHIR store containing the concept
      map(s) to use for the translation.
    source: The source value set of the concept map to be used. If unset,
      target is used to search for concept maps.
    system: Required. The system for the code to be translated.
    target: The target value set of the concept map to be used. If unset,
      source is used to search for concept maps.
    url: The canonical url of the concept map to use. If unset, the source and
      target is used to search for concept maps.
  """
    code = _messages.StringField(1)
    conceptMapVersion = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    source = _messages.StringField(4)
    system = _messages.StringField(5)
    target = _messages.StringField(6)
    url = _messages.StringField(7)