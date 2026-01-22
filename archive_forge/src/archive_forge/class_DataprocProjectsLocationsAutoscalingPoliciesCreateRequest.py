from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsAutoscalingPoliciesCreateRequest(_messages.Message):
    """A DataprocProjectsLocationsAutoscalingPoliciesCreateRequest object.

  Fields:
    autoscalingPolicy: A AutoscalingPolicy resource to be passed as the
      request body.
    parent: Required. The "resource name" of the region or location, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.autoscalingPolicies.create, the resource name of the
      region has the following format: projects/{project_id}/regions/{region}
      For projects.locations.autoscalingPolicies.create, the resource name of
      the location has the following format:
      projects/{project_id}/locations/{location}
  """
    autoscalingPolicy = _messages.MessageField('AutoscalingPolicy', 1)
    parent = _messages.StringField(2, required=True)