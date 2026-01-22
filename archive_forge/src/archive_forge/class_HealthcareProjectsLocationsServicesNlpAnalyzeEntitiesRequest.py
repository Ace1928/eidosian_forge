from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsServicesNlpAnalyzeEntitiesRequest(_messages.Message):
    """A HealthcareProjectsLocationsServicesNlpAnalyzeEntitiesRequest object.

  Fields:
    analyzeEntitiesRequest: A AnalyzeEntitiesRequest resource to be passed as
      the request body.
    nlpService: The resource name of the service of the form:
      "projects/{project_id}/locations/{location_id}/services/nlp".
  """
    analyzeEntitiesRequest = _messages.MessageField('AnalyzeEntitiesRequest', 1)
    nlpService = _messages.StringField(2, required=True)