from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsLocationsResourcesGeneratePackagesSummaryRequest(_messages.Message):
    """A
  ContaineranalysisProjectsLocationsResourcesGeneratePackagesSummaryRequest
  object.

  Fields:
    generatePackagesSummaryRequest: A GeneratePackagesSummaryRequest resource
      to be passed as the request body.
    name: Required. The name of the resource to get a packages summary for in
      the form of `projects/[PROJECT_ID]/resources/[RESOURCE_URL]`.
  """
    generatePackagesSummaryRequest = _messages.MessageField('GeneratePackagesSummaryRequest', 1)
    name = _messages.StringField(2, required=True)