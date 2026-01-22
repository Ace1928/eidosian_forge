from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsAutoscalingPoliciesGetRequest(_messages.Message):
    """A DataprocProjectsLocationsAutoscalingPoliciesGetRequest object.

  Fields:
    name: Required. The "resource name" of the autoscaling policy, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.autoscalingPolicies.get, the resource name of the
      policy has the following format:
      projects/{project_id}/regions/{region}/autoscalingPolicies/{policy_id}
      For projects.locations.autoscalingPolicies.get, the resource name of the
      policy has the following format: projects/{project_id}/locations/{locati
      on}/autoscalingPolicies/{policy_id}
  """
    name = _messages.StringField(1, required=True)