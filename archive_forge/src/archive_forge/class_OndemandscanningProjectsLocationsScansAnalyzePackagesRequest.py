from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OndemandscanningProjectsLocationsScansAnalyzePackagesRequest(_messages.Message):
    """A OndemandscanningProjectsLocationsScansAnalyzePackagesRequest object.

  Fields:
    analyzePackagesRequestV1: A AnalyzePackagesRequestV1 resource to be passed
      as the request body.
    parent: Required. The parent of the resource for which analysis is
      requested. Format: projects/[project_name]/locations/[location]
  """
    analyzePackagesRequestV1 = _messages.MessageField('AnalyzePackagesRequestV1', 1)
    parent = _messages.StringField(2, required=True)