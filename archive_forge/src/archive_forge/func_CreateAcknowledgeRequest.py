from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateAcknowledgeRequest(name, comment, acknowledge_type=None, release_track=ReleaseTrack.GA):
    """Construct an Assured Workload Violation Acknowledgement Request.

  Args:
    name: str, the name for the Assured Workloads violation being described in
      the form:
      organizations/{ORG_ID}/locations/{LOCATION}/workloads/{WORKLOAD_ID}/violations/{VIOLATION_ID}.
    comment: str, the business justification which the user wants to add while
      acknowledging a violation.
    acknowledge_type: str, the acknowledge type for specified violation, which
      is one of: SINGLE_VIOLATION - to acknowledge specified violation,
      EXISTING_CHILD_RESOURCE_VIOLATIONS - to acknowledge specified org policy
      violation and all associated child resource violations.
    release_track: ReleaseTrack, gcloud release track being used

  Returns:
    A populated Assured Workloads Violation Acknowledgement Request.
  """
    messages = util.GetMessagesModule(release_track)
    if acknowledge_type:
        acknowledge_type = messages.GoogleCloudAssuredworkloadsV1beta1AcknowledgeViolationRequest.AcknowledgeTypeValueValuesEnum(acknowledge_type)
    if release_track == ReleaseTrack.GA:
        return messages.AssuredworkloadsOrganizationsLocationsWorkloadsViolationsAcknowledgeRequest(googleCloudAssuredworkloadsV1AcknowledgeViolationRequest=messages.GoogleCloudAssuredworkloadsV1AcknowledgeViolationRequest(comment=comment), name=name)
    else:
        return messages.AssuredworkloadsOrganizationsLocationsWorkloadsViolationsAcknowledgeRequest(googleCloudAssuredworkloadsV1beta1AcknowledgeViolationRequest=messages.GoogleCloudAssuredworkloadsV1beta1AcknowledgeViolationRequest(comment=comment, acknowledgeType=acknowledge_type), name=name)