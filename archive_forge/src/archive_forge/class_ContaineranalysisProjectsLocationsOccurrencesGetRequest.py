from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsLocationsOccurrencesGetRequest(_messages.Message):
    """A ContaineranalysisProjectsLocationsOccurrencesGetRequest object.

  Fields:
    name: Required. The name of the occurrence in the form of
      `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]`.
  """
    name = _messages.StringField(1, required=True)