from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateCreateRequest(external_id, parent, workload, release_track=ReleaseTrack.GA):
    """Construct an Assured Workload Create Request for Assured Workloads API requests.

  Args:
    external_id: str, the identifier that identifies this Assured Workloads
      environment externally.
    parent: str, the parent organization of the Assured Workloads environment to
      be created, in the form: organizations/{ORG_ID}/locations/{LOCATION}.
    workload: Workload, new Assured Workloads environment containing the values
      to be used.
    release_track: ReleaseTrack, gcloud release track being used

  Returns:
    A populated Assured Workloads Update Request for the Assured Workloads API.
  """
    if release_track == ReleaseTrack.GA:
        return util.GetMessagesModule(release_track).AssuredworkloadsOrganizationsLocationsWorkloadsCreateRequest(externalId=external_id, parent=parent, googleCloudAssuredworkloadsV1Workload=workload)
    else:
        return util.GetMessagesModule(release_track).AssuredworkloadsOrganizationsLocationsWorkloadsCreateRequest(externalId=external_id, parent=parent, googleCloudAssuredworkloadsV1beta1Workload=workload)