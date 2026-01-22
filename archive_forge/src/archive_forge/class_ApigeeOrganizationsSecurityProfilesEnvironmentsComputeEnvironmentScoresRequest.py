from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesEnvironmentsComputeEnvironmentScoresRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesEnvironmentsComputeEnvironmentScore
  sRequest object.

  Fields:
    googleCloudApigeeV1ComputeEnvironmentScoresRequest: A
      GoogleCloudApigeeV1ComputeEnvironmentScoresRequest resource to be passed
      as the request body.
    profileEnvironment: Required. Name of organization and environment and
      profile id for which score needs to be computed. Format:
      organizations/{org}/securityProfiles/{profile}/environments/{env}
  """
    googleCloudApigeeV1ComputeEnvironmentScoresRequest = _messages.MessageField('GoogleCloudApigeeV1ComputeEnvironmentScoresRequest', 1)
    profileEnvironment = _messages.StringField(2, required=True)